"""Inspect page: pick a ready component, upload a file, see the verdict —
and, if reporting is enabled, the report once it's ready.

Pure view layer — detection logic is delegated to
emil_ml.core.inspections.orchestrator (which itself delegates to
pipeline.inspect); nothing here decides verdicts or generates reports.
The bounding-box overlay is driven entirely by whether `details` contains
a `detections` list (YOLO's shape); the heatmap overlay by whether it
contains a `heatmap` array (PatchCore's shape) — not by checking the
component's model_type — so this page still never branches on which
method produced the result.

The verdict/score/overlay show immediately (detection is fast); the
report (if applicable) is generated in a background thread and may take
up to a couple of minutes (see orchestrator.py) — this page never blocks
waiting for it. While pending, `_render_pending_report()` (an
`st.fragment(run_every=1)`) auto-refreshes just that section — live
stage messages plus the LLM's thinking/response streaming in token by
token, from core/inspections/progress.py's in-memory state — without
polling the rest of the page. It escalates to a full `st.rerun()` the
moment the background thread finishes, landing in the plain
complete/failed branches below on the very next full rerun.
"""

from __future__ import annotations

import io

import streamlit as st
from PIL import Image

from emil_ml.config.logging_config import configure_logging
from emil_ml.config.registry import ComponentRegistry
from emil_ml.core import registry_factory
from emil_ml.core.anomaly.patchcore.heatmap import render_heatmap_overlay
from emil_ml.core.detection.yolo.annotation import render_boxes_on_image
from emil_ml.core.inspections import orchestrator, progress, store

configure_logging()

st.set_page_config(page_title="EMIL Lab — Inspect", page_icon="🔍", layout="wide")
st.title("Inspect")

registry = ComponentRegistry()
# Cascade-only components (coco_detector) have no approved/failed verdict and
# no specialist dispatch on this path — see registry_factory.is_cascade_only().
# They run exclusively through Onboard's "Run the cascade" section.
all_ready = registry.list_ready()
ready_components = [c for c in all_ready if not registry_factory.is_cascade_only(c.model_type)]
cascade_only_components = [c for c in all_ready if registry_factory.is_cascade_only(c.model_type)]

if cascade_only_components:
    st.caption(
        "Object & face cascade component(s) — "
        + ", ".join(c.display_name for c in cascade_only_components)
        + " — aren't run from this page (no approved/failed verdict applies to them). "
        "Use **Onboard → Cascade: object & face recognition → Run the cascade** instead."
    )

if not ready_components:
    st.info("No trained components yet. Onboard and train one on the **Onboard** page first.")
    st.stop()

names = {c.display_name: c.name for c in ready_components}
selected_display = st.selectbox("Component", list(names.keys()))
component = registry.get(names[selected_display])

with st.sidebar:
    st.header("Session settings")
    inspector_name = st.text_input(
        "Inspector name",
        value="operator",
        help="Recorded as who ran each inspection you start below — shown on the Inspection "
        "Station as \"Run by\", separate from who later acknowledges it.",
    )
    override_threshold = st.checkbox("Override decision threshold for this session", value=False)
    threshold_value = None
    if override_threshold:
        # Every score this app produces (reconstruction error on [0,1]-normalized
        # images, a classifier's sigmoid probability, or a YOLO confidence) is
        # bounded to [0, 1], so this range works regardless of method.
        slider_max = min(max(component.anomaly_threshold * 3, 0.01), 1.0)
        slider_key = f"threshold_override_{component.name}"
        threshold_value = st.slider(
            "Threshold",
            min_value=0.0,
            max_value=slider_max,
            value=float(component.anomaly_threshold),
            format="%.5f",
            key=slider_key,
        )
        st.caption(
            "Lower catches more failures at the cost of more false alarms, and vice versa. "
            "By default this affects only verdicts shown in this session — the stored default "
            "is unchanged unless you save it below."
        )
        if st.button("Save as this component's default threshold"):
            registry.update_threshold(component.name, threshold_value)
            st.session_state.pop(slider_key, None)
            st.success(f"Saved {threshold_value:.5f} as the new default threshold.")
            st.rerun()

@st.fragment(run_every=1)
def _render_pending_report(inspection_id: int) -> None:
    """Auto-refreshing view of a report still being generated in the
    background — live stage messages plus the LLM's thinking/response
    text streaming in token by token (see core/inspections/progress.py).

    Ticks independently of the rest of the page (st.fragment) so this is
    the only thing that re-renders every second, not the whole upload/
    detection view above it. Escalates to a full `st.rerun()` — not
    `scope="fragment"` — the moment report_status stops being 'pending',
    which re-executes the whole script and lands in the plain
    complete/failed branches below instead of polling forever once done.
    """
    record = store.get(inspection_id)
    if record is None:
        return
    if record.report_status != "pending":
        st.rerun()
        return

    st.info(
        "⏳ Generating report... this can take up to a couple of minutes (the LLM reasons "
        "before answering). The verdict above is already final — this doesn't block on it."
    )
    state = progress.get(inspection_id)
    if state is not None:
        for stage in state.stages:
            st.caption(f"• {stage}")
        if state.thinking:
            with st.expander("Model reasoning (live)", expanded=True):
                st.text(state.thinking)
        if state.response:
            st.markdown("**Report (streaming in):**")
            st.markdown(state.response)


uploaded = st.file_uploader(
    "Drop or select a file to inspect", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
)

if uploaded is not None:
    # Keyed by Streamlit's own per-upload file_id, not just "an upload
    # exists" — so a NEW file triggers a fresh inspection, but re-running
    # this script for any OTHER reason (e.g. clicking "Check for
    # updates") reuses the same InspectionRecord instead of re-running
    # detection (and re-saving a second copy of the image) every time.
    session_key = f"inspect_{uploaded.file_id}"
    if session_key not in st.session_state:
        with st.spinner("Running detection..."):
            record, details = orchestrator.run_inspection(
                uploaded.getvalue(),
                component.name,
                threshold_override=threshold_value,
                registry=registry,
                run_by=inspector_name.strip() or "operator",
            )
        st.session_state[session_key] = {
            "inspection_id": record.id,
            "details": details,
            "image_bytes": uploaded.getvalue(),
        }

    state = st.session_state[session_key]
    # Re-fetched fresh from the DB on every rerun (not cached in
    # session_state) specifically so a completed background report shows
    # up without needing to re-run detection.
    record = store.get(state["inspection_id"])
    details = state["details"]
    image_bytes = state["image_bytes"]

    detections = details.get("detections")
    heatmap = details.get("heatmap")
    if detections:
        shown = [d for d in detections if d["confidence"] >= record.threshold] if record.threshold is not None else detections
        annotated = render_boxes_on_image(Image.open(io.BytesIO(image_bytes)), shown)
        st.image(annotated, caption=f"Input — {len(shown)} detection(s) shown", width=400)
    elif heatmap is not None:
        annotated = render_heatmap_overlay(Image.open(io.BytesIO(image_bytes)), heatmap)
        st.image(annotated, caption="Input — anomaly heatmap overlay (blue=normal, red=anomalous)", width=400)
    else:
        st.image(image_bytes, caption="Input", width=300)

    if record.verdict == "approved":
        st.success(f"APPROVED — score {record.score:.6f}" + (f" (threshold {record.threshold:.6f})" if record.threshold is not None else ""))
    else:
        st.error(f"FAILED — score {record.score:.6f}" + (f" (threshold {record.threshold:.6f})" if record.threshold is not None else ""))

    col1, col2 = st.columns(2)
    col1.metric("Score", f"{record.score:.6f}")
    col2.metric("Threshold", f"{record.threshold:.6f}" if record.threshold is not None else "—")

    if detections:
        st.caption("All candidate detections (including any below the active threshold):")
        st.json(detections)

    # --- Report ----------------------------------------------------------
    if record.report_status == "pending":
        st.divider()
        _render_pending_report(record.id)
    elif record.report_status == "complete":
        st.divider()
        st.subheader("Report")
        st.markdown(record.report_text)
        if record.machine_context_used:
            st.caption("**Machine context considered:** " + ", ".join(record.machine_context_used))
        if record.report_sources:
            with st.expander(f"Sources cited ({len(record.report_sources)})"):
                for s in record.report_sources:
                    st.markdown(f"- **{s['source']}** — {s['section']} (`{s['doc_type']}`)")
                    # .get(): reports persisted before source paths were added won't have this key.
                    if s.get("path"):
                        st.caption(s["path"])
        if record.report_prompt or record.report_thinking:
            with st.expander("LLM details (prompt & reasoning)"):
                if record.report_model:
                    st.caption(f"Model: `{record.report_model}`")
                if record.report_thinking:
                    st.markdown("**Model reasoning (raw):**")
                    st.text(record.report_thinking)
                if record.report_prompt:
                    st.markdown("**Exact prompt sent to the LLM:**")
                    st.code(record.report_prompt, language=None)
    elif record.report_status == "failed":
        st.divider()
        st.warning(record.report_text)
    # report_status == "none": reporting isn't enabled/applicable for this
    # inspection — nothing shown, same as before RAG existed.
