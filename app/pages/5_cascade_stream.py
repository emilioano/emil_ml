"""Cascade Stream page: continuous operation of the object/face cascade —
a live Kafka topic (via the standalone emil_ml.cascade_stream process,
started from a terminal — see this page's own "Live status" section), an
uploaded video file, or a single uploaded still image (both processed
synchronously, right here).

All three paths share core/cascade/stream_processor.py's per-frame throttle
+ dispatch logic and core/cascade/stream_store.py's persistence — this page
never re-implements cascade dispatch itself, only drives it.

Deliberately does NOT launch or manage the Kafka consumer process (no
subprocess/PID tracking) — same reasoning the folder watcher
(emil_ml.watcher) is never started from inside Streamlit: a session
restarts on every code change/rerun, which would kill an in-process
consumer. This page configures per-component settings and shows live
status/results by polling the database instead.

Onboard's own "Cascade: object & face recognition" section is unchanged and
still the place to create a component, register consenting individuals, and
set reaction policies, plus a one-shot single-image test run. This page is
purely about continuous operation of an already-configured component.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from emil_ml.config.logging_config import configure_logging
from emil_ml.config.registry import ComponentRegistry
from emil_ml.config.settings import (
    CASCADE_STREAM_HEARTBEAT_STALE_SECONDS,
    DEFAULT_CASCADE_STREAM_UI_POLL_SECONDS,
)
from emil_ml.core import registry_factory
from emil_ml.core.cascade import stream_processor, stream_store
from emil_ml.core.cascade.frame_sources import Frame, VideoFileFrameSource

configure_logging()

st.set_page_config(page_title="EMIL Lab — Cascade Stream", page_icon="📡", layout="wide")
st.title("Cascade Stream")
st.caption(
    "Run the object & face cascade continuously against a Kafka topic, on an uploaded video "
    "file, or on a single uploaded still image — instead of only testing one image at a time on "
    "the Onboard page."
)

_HEARTBEAT_FMT = "%Y-%m-%d %H:%M:%S"  # SQLite's own datetime('now') format
_VIDEO_TYPES = ["mp4", "avi", "mov", "mkv"]
_IMAGE_TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]  # matches app/pages/2_onboard.py's UPLOAD_TYPES


def _render_result_card(target, result: "stream_store.CascadeStreamResult") -> None:
    """Renders one processed-frame result — used both by the settled
    "Recent results" feed below (target=st, appended top-to-bottom, newest
    first) and by the live single-slot preview during video processing
    (target=an st.empty() placeholder, so each call REPLACES the previous
    frame shown there in place rather than stacking a new one below it —
    see the "Video file" section for why that distinction matters)."""
    with target.container(border=True):
        result_col1, result_col2 = st.columns([1, 2])
        if result.thumbnail_path:
            result_col1.image(result.thumbnail_path, width=200)
        result_col1.caption(
            f"{result.source} · {result.frame_ref} · {result.created_at} · {result.object_count} object(s)"
        )
        if not result.objects:
            result_col2.caption("No objects detected in this frame.")
        for obj in result.objects:
            result_col2.markdown(f"**{obj['label']}**")
            result_col2.caption(f"category: {obj['category']} · confidence: {obj['confidence']:.2f}")
            specialist_result = obj.get("specialist_result")
            if specialist_result is None:
                result_col2.caption("No specialist activated for this category.")
            else:
                if specialist_result["matched"]:
                    result_col2.success(f"Recognized: **{specialist_result['identity_label']}**")
                else:
                    reason = (specialist_result.get("details") or {}).get("reason", "no match")
                    result_col2.warning(f"Unknown person ({reason})")
                policy_result = obj.get("policy_result")
                if policy_result is not None:
                    policy = policy_result["policy"]
                    result_col2.markdown(f"**{policy['label']}** — _{policy['message']}_")
                    result_col2.caption(
                        f"actions triggered: {', '.join(policy_result['executed_actions']) or '(none)'} · "
                        f"priority: {policy['priority']}"
                    )

registry = ComponentRegistry()
cascade_components = [
    c for c in registry.list_active() if registry_factory.is_cascade_only(c.model_type) and c.status == "ready"
]

if not cascade_components:
    st.info(
        "No ready cascade components yet. Create an 'Object & face cascade (COCO detector)' "
        "component on the **Onboard** page first."
    )
    st.stop()

names = {c.display_name: c.name for c in cascade_components}
selected_display = st.selectbox("Component", list(names.keys()), key="cascade_stream_component_select")
component = registry.get(names[selected_display])

# --- Settings ----------------------------------------------------------------
with st.expander("Kafka & sampling settings", expanded=not component.cascade_stream_kafka_topic):
    st.caption(
        "The Kafka topic this component's live stream consumes (one message = one frame, raw "
        "image bytes), and how often a frame is actually run through the cascade — checking "
        "every single frame is rarely necessary."
    )
    new_bootstrap_servers = st.text_input(
        "Kafka bootstrap servers",
        value=component.cascade_stream_kafka_bootstrap_servers,
        placeholder="e.g. localhost:9092",
        key=f"cascade_stream_kafka_bootstrap_{component.name}",
    )
    new_topic = st.text_input(
        "Kafka topic",
        value=component.cascade_stream_kafka_topic,
        key=f"cascade_stream_kafka_topic_{component.name}",
    )
    new_sample_rate = st.number_input(
        "Check at most one frame every N seconds",
        min_value=0.1,
        value=float(component.cascade_stream_sample_rate_seconds),
        step=0.1,
        key=f"cascade_stream_sample_rate_{component.name}",
        help="Applies to both the Kafka stream and an uploaded video's own timeline — not how "
        "fast a video happens to decode.",
    )
    if st.button("Save settings", key=f"save_cascade_stream_settings_{component.name}"):
        registry.update_settings(
            component.name,
            cascade_stream_kafka_bootstrap_servers=new_bootstrap_servers.strip(),
            cascade_stream_kafka_topic=new_topic.strip(),
            cascade_stream_sample_rate_seconds=float(new_sample_rate),
        )
        st.success("Saved.")
        st.rerun()

# --- Live status (Kafka) ------------------------------------------------------
st.markdown("#### Live stream (Kafka)")

if not component.cascade_stream_kafka_bootstrap_servers or not component.cascade_stream_kafka_topic:
    st.warning("Configure Kafka bootstrap servers and a topic above first.")
else:
    active_run = stream_store.get_active_run(component.name)
    is_stale = True
    if active_run is not None:
        last_heartbeat = datetime.strptime(active_run.last_heartbeat_at, _HEARTBEAT_FMT).replace(tzinfo=timezone.utc)
        age_seconds = (datetime.now(timezone.utc) - last_heartbeat).total_seconds()
        is_stale = age_seconds > CASCADE_STREAM_HEARTBEAT_STALE_SECONDS

    if active_run is not None and not is_stale:
        st.success(
            f"Running — {active_run.frames_processed} frame(s) processed of "
            f"{active_run.frames_seen} seen, started {active_run.started_at}."
        )
    elif active_run is not None and is_stale:
        st.warning(
            f"Marked 'running' but no heartbeat for over {CASCADE_STREAM_HEARTBEAT_STALE_SECONDS:.0f}s — "
            "likely stopped or crashed without updating its status. "
            f"Last seen: {active_run.last_heartbeat_at}."
        )
    else:
        last_runs = stream_store.list_runs(component.name, limit=1)
        if last_runs and last_runs[0].status == "crashed":
            st.error(f"Last run crashed: {last_runs[0].last_error}")
        else:
            st.caption("Not currently running.")

    st.caption(
        "Starting and stopping the stream is a terminal action, not a button here — same reason "
        "the folder watcher is never launched from inside Streamlit: this page's own process "
        "restarts on every code change, which would kill a consumer running inside it."
    )
    st.code(f"python -m emil_ml.cascade_stream --component {component.name}", language="bash")

# --- Video file ---------------------------------------------------------------
st.markdown("#### Video file")
st.caption(
    "Process an uploaded video file's frames through the cascade, right now, at the sample rate "
    "above. While it's running, the most recently processed frame refreshes in place below — it "
    "isn't added to a growing list until processing finishes, so you can watch it work through "
    "the video one sampled frame at a time without the page filling up mid-run."
)

video_upload = st.file_uploader("Video file", type=_VIDEO_TYPES, key=f"cascade_stream_video_upload_{component.name}")
if video_upload is not None and st.button("Process video", key=f"cascade_stream_process_video_{component.name}"):
    suffix = Path(video_upload.name).suffix or ".mp4"
    tmp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_path = Path(tmp_file.name)
    try:
        tmp_file.write(video_upload.getvalue())
        tmp_file.close()

        source = VideoFileFrameSource(tmp_path)
        try:
            total_frames = source.frame_count()
            progress_bar = st.progress(0.0) if total_frames else None
            status_text = st.empty()
            live_preview = st.empty()  # refreshed in place per processed frame, cleared once done — see caption above

            run = stream_store.start_run(component.name, source="video", source_detail=video_upload.name)
            last_processed_position: float | None = None
            frames_seen = 0
            frames_processed = 0
            status = "completed"
            error: str | None = None
            try:
                for frame in source.frames():
                    frames_seen += 1
                    if stream_processor.should_sample(
                        last_processed_position, component.cascade_stream_sample_rate_seconds, frame.position_seconds
                    ):
                        result = stream_processor.process_frame(
                            frame, component, run_id=run.id, source="video", registry=registry
                        )
                        last_processed_position = frame.position_seconds
                        frames_processed += 1
                        _render_result_card(live_preview, result)
                    if progress_bar is not None:
                        progress_bar.progress(min(frames_seen / total_frames, 1.0))
                    status_text.caption(f"{frames_seen} frame(s) seen, {frames_processed} processed...")
            except Exception as exc:  # noqa: BLE001 - must still record the failure below, then surface it
                status = "crashed"
                error = str(exc)
                raise
            finally:
                stream_store.heartbeat(run.id, frames_seen=frames_seen, frames_processed=frames_processed)
                stream_store.finish_run(run.id, status=status, error=error)
                status_text.empty()
                # The single live-preview slot's job ends here — "Recent
                # results" below already has every processed frame (it's
                # been in the database since each process_frame() call
                # above) and takes over as the permanent, settled,
                # top-to-bottom list per the page's own docstring.
                live_preview.empty()
        finally:
            source.close()

        st.success(f"Done — {frames_processed} of {frames_seen} frame(s) processed.")
    finally:
        tmp_path.unlink(missing_ok=True)

# --- Still image ---------------------------------------------------------------
st.markdown("#### Still image")
st.caption("Run the cascade once against a single uploaded image — no throttling, always processed.")

image_upload = st.file_uploader("Image", type=_IMAGE_TYPES, key=f"cascade_stream_image_upload_{component.name}")
if image_upload is not None and st.button("Process image", key=f"cascade_stream_process_image_{component.name}"):
    frame = Frame(
        raw=image_upload.getvalue(),
        frame_ref=f"upload@{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}",
        position_seconds=0.0,
        received_at=datetime.now(timezone.utc),
    )
    run = stream_store.start_run(component.name, source="image", source_detail=image_upload.name)
    try:
        result = stream_processor.process_frame(frame, component, run_id=run.id, source="image", registry=registry)
        stream_store.heartbeat(run.id, frames_seen=1, frames_processed=1)
        stream_store.finish_run(run.id, status="completed")
    except Exception as exc:  # noqa: BLE001 - must still record the failure below, then surface it
        stream_store.finish_run(run.id, status="crashed", error=str(exc))
        raise
    st.success(f"Done — {result.object_count} object(s) detected.")
    _render_result_card(st, result)

# --- Live results feed --------------------------------------------------------
feed_header_col1, feed_header_col2 = st.columns([3, 1])
feed_header_col1.markdown("#### Recent results")
if feed_header_col2.button(
    "🧹 Clear results & images",
    key=f"cascade_stream_clean_{component.name}",
    help="Deletes every stored result and its thumbnail image for this component. Run history "
    "(when streams ran, how many frames) is kept — this only clears the per-frame images.",
):
    removed = stream_store.delete_results_for_component(component.name)
    st.success(f"Removed {removed} result(s) and their images.")
    st.rerun()


@st.fragment(run_every=DEFAULT_CASCADE_STREAM_UI_POLL_SECONDS)
def _render_recent_results(component_name: str) -> None:
    results = stream_store.list_recent_results(component_name, limit=50)
    if not results:
        st.caption("No frames processed yet for this component.")
        return
    for result in results:
        _render_result_card(st, result)


_render_recent_results(component.name)
