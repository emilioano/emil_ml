"""Inspection Station (formerly "History"): a review workstation built
around the fact that approved inspections are noise most of the time and
failed ones need a human's attention — but an approved record must never
become unreachable, since it could be a false negative (see
config/settings.py's VALID_APPROVED_HANDLING_MODES for the full reasoning).

Two independent concerns, deliberately kept separate:

- FILTERING decides which records are visible at all (component, lifecycle
  status, report status, and whether approved records are shown). This is
  the only thing that can make a record disappear from the list —
  everything filtered out is still in the database, just not fetched/shown
  here.
- SORTING decides the order of whatever filtering left visible. It never
  changes which records show, only how they're arranged — "failed first"
  is one sort mode among a few, not a hidden default someone has to
  discover is happening. The station's own default is plain chronological
  (newest first), the least surprising choice for what is also a history
  view; an operator opting into review mode picks "Failed first" themselves.

Every filter AND the sort mode round-trip through st.query_params, so a
station can bookmark a URL that opens pre-filtered and pre-sorted (e.g.
"only tandborste, only failed verdicts, failed-first"). Reading is
deliberately forgiving: an unrecognized or missing parameter silently
falls back to its default rather than raising — a stale or hand-edited
URL degrades to the normal view instead of crashing the page.

Pure view layer over core.inspections.store/lifecycle, same as before this
rewrite: acknowledging only flips a status flag (files untouched);
archiving is a separate, explicit action that moves files to a
date-partitioned archive with a DB-atomic path update. Nothing here
deletes anything, ever — reverting acknowledged -> new is itself just
another status flag flip, not a file operation (see store.revert_to_new()).
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

# See app/pages/2_onboard.py's identical bootstrap comment — shared_yolo_canvas.py
# lives in app/ (a sibling of pages/), not in the emil_ml package.
_APP_DIR = Path(__file__).resolve().parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from shared_yolo_canvas import render_yolo_box_canvas  # noqa: E402 (after sys.path bootstrap above)

from emil_ml.config.logging_config import configure_logging
from emil_ml.config.registry import ComponentRegistry
from emil_ml.core.inspections import lifecycle, retention, store
from emil_ml.training import onboard
from emil_ml.utils.paths import for_component

configure_logging()

st.set_page_config(page_title="EMIL Lab — Inspection Station", page_icon="🗂️", layout="wide")
st.title("Inspection Station")
st.caption("Review and acknowledge inspections — formerly \"History\".")
st.caption(
    "Lifecycle: **new → acknowledged** (kvittering — a human has dealt with it; automatic for "
    "approved records set to auto_acknowledge) → **acknowledged → archived** (arkivering — files "
    "moved to a date-partitioned archive/, one at a time or in bulk below; never for failed "
    "records) → **archived → deleted** (governed by each component's \"Keep archived inspections "
    "for N days\" setting on the Onboard page — not enforced automatically yet, and unrelated to "
    "when a record gets acknowledged or archived in the first place)."
)

if "_bulk_result_messages" in st.session_state:
    for kind, message in st.session_state.pop("_bulk_result_messages"):
        getattr(st, kind)(message)

registry = ComponentRegistry()
all_components = registry.list_all()

_pending_verified = retention.pending_verified_counts(registry=registry)
if _pending_verified:
    _total_pending = sum(_pending_verified.values())
    _by_component = ", ".join(f"{name} ({count})" for name, count in sorted(_pending_verified.items()))
    st.info(
        f"⏳ {_total_pending} verified correction(s) waiting for a retrain to incorporate them "
        f"— {_by_component}. These are protected from retention deletion until then (see the "
        "Onboard page's Train section to incorporate them)."
    )

SORT_MODES = ("chronological", "failed_first", "score", "status")
SORT_LABELS = {
    "chronological": "Newest first (chronological)",
    "failed_first": "Failed first (review mode)",
    "score": "Score (highest first)",
    "status": "Status (new, then acknowledged, then archived)",
}
DEFAULT_SORT = "chronological"

FETCH_LIMIT = 500  # generous headroom so Python-side approved-hiding never starves the display cap below
DISPLAY_LIMIT = 200


def _bool_param(name: str, default: bool) -> bool:
    raw = st.query_params.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# --- Read + validate every filter/sort param up front, all silently -------
# falling back to a default on anything missing or unrecognized — a
# bookmarked URL from a deleted component, an old param name, or a typo
# should degrade to the normal view, never crash the page.

component_slugs = {c.name for c in all_components}
raw_component = st.query_params.get("component")
initial_component = raw_component if raw_component in component_slugs else "all"

raw_status = st.query_params.get("status")
raw_verdict = st.query_params.get("verdict")
initial_verdict = raw_verdict if raw_verdict in ("approved", "failed") else "all"
initial_status = raw_status if raw_status in store.VALID_STATUSES else "all"
if initial_status == "all" and initial_verdict == "all" and raw_status in ("approved", "failed"):
    # Forgiving alias: "?status=failed" reads naturally as "show failed
    # verdicts" even though "failed" isn't a lifecycle status (those are
    # new/acknowledged/archived) — honor the evident intent rather than
    # silently dropping it just because it landed in the wrong-named param.
    initial_verdict = raw_status

raw_report_status = st.query_params.get("report_status")
initial_report_status = raw_report_status if raw_report_status in store.VALID_REPORT_STATUSES else "all"

raw_verified = st.query_params.get("verified")
initial_verified = raw_verified if raw_verified in store.VALID_VERIFIED_STATUSES else "all"

initial_show_approved = _bool_param("show_approved", False)

raw_sort = st.query_params.get("sort")
initial_sort = raw_sort if raw_sort in SORT_MODES else DEFAULT_SORT

with st.sidebar:
    st.header("Filters")
    st.caption("Which records are visible at all.")

    component_options = ["all", *sorted(component_slugs)]
    display_by_slug = {c.name: c.display_name for c in all_components}
    component_choice = st.selectbox(
        "Component",
        component_options,
        index=component_options.index(initial_component),
        format_func=lambda s: "All components" if s == "all" else display_by_slug.get(s, s),
        key="filter_component",
    )

    verdict_options = ["all", "failed", "approved"]
    verdict_choice = st.selectbox(
        "Verdict",
        verdict_options,
        index=verdict_options.index(initial_verdict),
        format_func=lambda v: {"all": "All verdicts", "failed": "❌ Failed", "approved": "✅ Approved"}[v],
        key="filter_verdict",
    )

    status_options = ["all", *store.VALID_STATUSES]
    status_choice = st.selectbox(
        "Lifecycle status",
        status_options,
        index=status_options.index(initial_status),
        format_func=lambda s: "All statuses" if s == "all" else s,
        key="filter_status",
    )

    report_status_options = ["all", *store.VALID_REPORT_STATUSES]
    report_status_choice = st.selectbox(
        "Report status",
        report_status_options,
        index=report_status_options.index(initial_report_status),
        format_func=lambda s: "All" if s == "all" else s,
        key="filter_report_status",
    )

    verified_options = ["all", *store.VALID_VERIFIED_STATUSES]
    verified_choice = st.selectbox(
        "Verification",
        verified_options,
        index=verified_options.index(initial_verified),
        format_func=lambda v: {
            "all": "All",
            "unverified": "⬜ Unverified",
            "verified_correct": "✅ Verified correct",
            "verified_incorrect": "🚩 Verified incorrect (flagged)",
        }[v],
        key="filter_verified",
        help="Combinable with every filter above — e.g. Verdict=Approved + "
        "Verification=Unverified surfaces exactly the hidden false-negative candidates "
        "worth reviewing.",
    )

    show_approved_choice = st.checkbox(
        "Show approved (when Verdict = All)",
        value=initial_show_approved,
        key="filter_show_approved",
        help="Approved records are hidden from the default view (never deleted — always "
        "reachable here) unless a component's own 'Approved verdicts' setting is 'Keep "
        "visible' (see Onboard page). This only matters when Verdict is set to 'All' above; "
        "explicitly filtering Verdict = Approved always shows them regardless of this box.",
    )

    st.divider()
    st.header("Sort")
    st.caption("Order of whatever the filters above left visible — independent of filtering.")
    sort_choice = st.selectbox(
        "Order by",
        SORT_MODES,
        index=SORT_MODES.index(initial_sort),
        format_func=lambda s: SORT_LABELS[s],
        key="filter_sort",
    )

    st.divider()
    acknowledged_by = st.text_input(
        "Acknowledged by", value="operator", help="Recorded against each inspection you acknowledge below."
    )

# --- Keep the URL in sync with whatever's currently selected, so the ------
# current view is always what gets bookmarked/shared, whether it got here
# via the URL or via the controls above.
st.query_params["component"] = component_choice
st.query_params["verdict"] = verdict_choice
st.query_params["status"] = status_choice
st.query_params["report_status"] = report_status_choice
st.query_params["verified"] = verified_choice
st.query_params["show_approved"] = "1" if show_approved_choice else "0"
st.query_params["sort"] = sort_choice

component_filter = None if component_choice == "all" else component_choice
verdict_filter = None if verdict_choice == "all" else verdict_choice
status_filter = None if status_choice == "all" else status_choice
report_status_filter = None if report_status_choice == "all" else report_status_choice
verified_filter = None if verified_choice == "all" else verified_choice

# --- Bulk selection lives in session_state, reset whenever the FILTER ----
# changes (never on a sort change — sort only reorders the same visible
# set, it never changes which records a stale selection would silently
# still point at). Deliberately simpler than trying to carry a selection
# across a filter change: a selection made against one filtered view
# generally shouldn't silently keep acting once the visible set itself
# has changed.
_filter_signature = (
    component_choice, verdict_choice, status_choice, report_status_choice, verified_choice, show_approved_choice
)
if st.session_state.get("_last_filter_signature") != _filter_signature:
    for _key in [k for k in st.session_state if k.startswith("sel_")]:
        del st.session_state[_key]
    st.session_state.pop("select_all_visible", None)
    st.session_state["_last_filter_signature"] = _filter_signature

inspections = store.list_all(
    component_name=component_filter,
    status=status_filter,
    verdict=verdict_filter,
    report_status=report_status_filter,
    verified_status=verified_filter,
    limit=FETCH_LIMIT,
)

if verdict_filter is None and not show_approved_choice:
    keep_visible_components = {c.name for c in all_components if c.approved_handling == "keep_visible"}
    inspections = [
        r for r in inspections if r.verdict != "approved" or r.component_name in keep_visible_components
    ]

inspections = inspections[:DISPLAY_LIMIT]

if sort_choice == "failed_first":
    inspections = sorted(inspections, key=lambda r: 0 if r.verdict == "failed" else 1)
elif sort_choice == "score":
    inspections = sorted(inspections, key=lambda r: r.score, reverse=True)
elif sort_choice == "status":
    _status_rank = {"new": 0, "acknowledged": 1, "archived": 2}
    inspections = sorted(inspections, key=lambda r: _status_rank.get(r.status, 99))
# "chronological" needs no re-sort — store.list_all() already returns newest-first.

if not inspections:
    st.info("No inspections match these filters yet.")
    st.stop()

st.caption(
    f"{len(inspections)} inspection(s) shown, sorted by: {SORT_LABELS[sort_choice]} "
    f"(capped at {DISPLAY_LIMIT})."
)


@st.dialog("Confirm bulk acknowledge")
def _open_bulk_acknowledge_dialog(selected_ids: list[int], by: str) -> None:
    st.write(f"Acknowledge {len(selected_ids)} selected inspection(s)?")
    st.caption("Only a status flag flips (new → acknowledged) — no files move.")
    bcol1, bcol2 = st.columns(2)
    if bcol1.button(f"Acknowledge {len(selected_ids)}", key="confirm_bulk_ack", type="primary"):
        count = store.bulk_acknowledge(selected_ids, by=by or "operator")
        for rid in selected_ids:
            st.session_state.pop(f"sel_{rid}", None)
        st.session_state.pop("select_all_visible", None)
        st.session_state["_bulk_result_messages"] = [("success", f"Acknowledged {count} inspection(s).")]
        st.rerun()
    if bcol2.button("Cancel", key="cancel_bulk_ack"):
        st.rerun()


@st.dialog("Confirm bulk archive")
def _open_bulk_archive_dialog(selected_ids: list[int], by: str) -> None:
    records = [r for r in (store.get(i) for i in selected_ids) if r is not None]
    approved = [r for r in records if r.verdict == "approved" and r.status != "archived"]
    failed_count = sum(1 for r in records if r.verdict == "failed")
    already_archived_count = sum(1 for r in records if r.status == "archived")

    st.write(f"Archive {len(approved)} selected approved record(s)?")
    st.caption(
        "Moves files into a date-partitioned archive/ folder and updates the database. "
        "Archiving never deletes anything — an archived record stays fully searchable, "
        "e.g. if a suspected false negative among approved records needs to be dug back up later."
    )
    if failed_count:
        st.warning(
            f"{failed_count} failed record(s) in your selection will be SKIPPED — failed records "
            "can never be bulk-archived, they always require individual review first."
        )
    if already_archived_count:
        st.caption(f"{already_archived_count} already archived — will be skipped.")

    if not approved:
        st.info("Nothing left to archive in this selection.")
        if st.button("Close", key="close_bulk_archive_empty"):
            st.rerun()
        return

    bcol1, bcol2 = st.columns(2)
    if bcol1.button(f"Archive {len(approved)} record(s)", key="confirm_bulk_archive", type="primary"):
        progress_bar = st.progress(0.0, text="Archiving...")

        def _on_progress(done: int, total: int) -> None:
            progress_bar.progress(done / total, text=f"Archiving {done}/{total}...")

        result = lifecycle.bulk_archive_approved([r.id for r in approved], by=by or "operator", on_progress=_on_progress)

        for rid in selected_ids:
            st.session_state.pop(f"sel_{rid}", None)
        st.session_state.pop("select_all_visible", None)

        messages: list[tuple[str, str]] = [("success", f"Archived {result.archived} inspection(s).")]
        if result.skipped_failed_verdict:
            messages.append(
                ("warning", f"Skipped {result.skipped_failed_verdict} failed record(s) — never bulk-archived.")
            )
        if result.skipped_already_archived:
            messages.append(("caption", f"Skipped {result.skipped_already_archived} already-archived record(s)."))
        if result.errors:
            messages.append(("error", f"{len(result.errors)} record(s) could not be archived: {result.errors[:5]}"))
        st.session_state["_bulk_result_messages"] = messages
        st.rerun()
    if bcol2.button("Cancel", key="cancel_bulk_archive"):
        st.rerun()


visible_ids = [r.id for r in inspections]


def _on_select_all_change() -> None:
    new_value = st.session_state["select_all_visible"]
    for rid in visible_ids:
        st.session_state[f"sel_{rid}"] = new_value


st.checkbox(
    f"Select all {len(visible_ids)} visible",
    key="select_all_visible",
    on_change=_on_select_all_change,
    help="Ticks every currently visible (filtered) record in one go — you can still uncheck "
    "individual ones afterward. A shortcut into an adjustable selection, not a separate action.",
)

selected_ids = [rid for rid in visible_ids if st.session_state.get(f"sel_{rid}", False)]

bulk_col1, bulk_col2, bulk_col3 = st.columns([1, 1, 3])
with bulk_col1:
    if st.button(f"Acknowledge selected ({len(selected_ids)})", disabled=not selected_ids, key="open_bulk_ack"):
        _open_bulk_acknowledge_dialog(selected_ids, acknowledged_by)
with bulk_col2:
    if st.button(f"Archive selected ({len(selected_ids)})", disabled=not selected_ids, key="open_bulk_archive"):
        _open_bulk_archive_dialog(selected_ids, acknowledged_by)
with bulk_col3:
    st.caption("Bulk actions act only on checked rows below — never on everything the filter shows.")

st.divider()

component_by_name = {c.name: c for c in all_components}
_yolo_class_cache: dict[str, list[str]] = {}


def _yolo_classes_for(component_name: str) -> list[str]:
    if component_name not in _yolo_class_cache:
        _yolo_class_cache[component_name] = onboard.get_yolo_classes(component_name)
    return _yolo_class_cache[component_name]


VERIFIED_BADGES = {
    "unverified": "⬜ Unverified",
    "verified_correct": "✅ Verified correct",
    "verified_incorrect": "🚩 Flagged incorrect",
}

# Self-explanatory verification choices — "correct" deliberately means
# the MODEL'S RESULT matched reality, not "the unit was approved": a
# failed verdict that correctly caught a real defect is just as much
# "correct" as an approved verdict that correctly found nothing. Offering
# this only for approved records would skew the verified pool toward
# nothing but confirmed-approved examples, wrecking both accuracy
# measurement and retraining data (see training/onboard.py's
# incorporate_verified_corrections(), which needs real confirmed-
# failed examples too, not just confirmed-approved ones).
VERIFY_ACTIONS = {
    "correct": {
        "label": "✅ Prediction was correct",
        "explain": "The model's result matches reality — whether it said approved or failed, it got this one right.",
    },
    "false_positive": {
        "label": "🚩 Wrong — this was actually OK (false positive)",
        "explain": "The model flagged a defect, but the product is actually fine. Recorded as a confirmed-good "
        "example — no drawing needed.",
    },
    "false_negative": {
        "label": "🚩 Wrong — a defect was missed (false negative)",
        "explain": "The model approved this, but it actually has a real defect. The most dangerous kind of "
        "miss — you'll mark where the defect is.",
    },
    "wrong_class": {
        "label": "🚩 Wrong — right area, wrong type (misclassification)",
        "explain": "The model found something real but called it the wrong kind of defect. You'll correct the "
        "type (and adjust the box if needed).",
    },
}

for record in inspections:
    paths = for_component(record.component_name)
    cols = st.columns([0.4, 1, 2, 3, 1.5])

    with cols[0]:
        # No `value=` here on purpose: this key is sometimes already in
        # session_state (set by _on_select_all_change() above, or by the
        # widget's own prior render) — passing `value=` alongside an
        # existing session_state entry for the same key is redundant and
        # triggers a Streamlit warning; the key alone is enough for the
        # checkbox to pick up and persist its own state.
        st.checkbox("Select", key=f"sel_{record.id}", label_visibility="collapsed")

    with cols[1]:
        image_abs = paths.root / record.image_path if record.image_path else None
        if image_abs and image_abs.exists():
            st.image(str(image_abs), width=100)
        else:
            st.caption("(image unavailable)")

    with cols[2]:
        st.markdown(f"**{record.component_name}**")
        verdict_emoji = "✅" if record.verdict == "approved" else "❌"
        st.markdown(f"{verdict_emoji} {record.verdict} — score {record.score:.4f}")
        if record.defect_classes:
            st.caption(", ".join(record.defect_classes))
        st.caption(record.created_at)
        st.caption(f"Run by: {record.run_by or '—'}")
        verified_line = VERIFIED_BADGES[record.verified_status]
        if record.verified_status == "verified_incorrect" and record.verified_error_type:
            verified_line += f" ({record.verified_error_type.replace('_', ' ')})"
        st.caption(verified_line)
        if record.verified_status != "unverified" and record.verified_by:
            st.caption(f"Verified by: {record.verified_by}")
        if record.verified_status != "unverified":
            if st.button("↩ Undo verification", key=f"unverify_{record.id}", help="Clears this verification — for a misclick (wrong flag, wrong button). Safe to undo any time before a retrain has consumed it."):
                store.unverify(record.id)
                st.rerun()

    with cols[3]:
        status_badge = {"new": "🆕 new", "acknowledged": "👁️ acknowledged", "archived": "🗄️ archived"}[record.status]
        st.markdown(status_badge)
        if record.status in ("acknowledged", "archived") and record.acknowledged_by:
            st.caption(f"Acknowledged by: {record.acknowledged_by}")
        report_badge = {
            "none": "No report",
            "pending": "⏳ Report generating...",
            "complete": "📄 Report available",
            "failed": "⚠️ Report generation failed",
        }[record.report_status]
        st.caption(report_badge)
        if record.report_status == "failed" and record.report_text:
            # report_text is still a real, honest error message here (see
            # orchestrator.py's report_status='failed' handling) — surfaced
            # directly under the badge rather than requiring a click into
            # an expander that looked, until now, exactly like a normal
            # report ("View report" only ever appeared for report_status
            # == 'complete').
            st.caption(record.report_text)
        if record.report_status == "complete" and record.report_text:
            with st.expander("View report"):
                st.markdown(record.report_text)
                if record.report_sources:
                    st.caption(
                        "Sources: "
                        + "; ".join(
                            f"{s['source']} / {s['section']}"
                            + (f" ({s['path']})" if s.get("path") else "")  # .get(): older persisted reports lack this key
                            for s in record.report_sources
                        )
                    )
                if record.report_prompt or record.report_thinking:
                    # Streamlit can't nest an expander inside "View report"
                    # itself an expander — shown inline, gated behind a
                    # checkbox instead so it stays out of the way by default.
                    if st.checkbox("Show LLM details (prompt & reasoning)", key=f"llm_details_{record.id}"):
                        if record.report_model:
                            st.caption(f"Model: `{record.report_model}`")
                        if record.report_thinking:
                            st.markdown("**Model reasoning (raw):**")
                            st.text(record.report_thinking)
                        if record.report_prompt:
                            st.markdown("**Exact prompt sent to the LLM:**")
                            st.code(record.report_prompt, language=None)

    with cols[4]:
        if record.status == "new":
            if st.button("Acknowledge", key=f"ack_{record.id}"):
                store.acknowledge(record.id, by=acknowledged_by or "operator")
                st.rerun()
        elif record.status == "acknowledged":
            if st.button("Archive", key=f"archive_{record.id}"):
                new_image_path, new_report_path = lifecycle.archive(paths, record)
                store.mark_archived(record.id, image_path=new_image_path, report_path=new_report_path)
                st.rerun()
            if st.button("Revert to new", key=f"revert_{record.id}"):
                store.revert_to_new(record.id)
                st.rerun()
        else:
            st.caption(f"Archived {record.archived_at}")
            st.caption("Files moved — cannot be reverted here.")

    # --- Verify / flag: full width, not squeezed into a column — the -----
    # correction canvas below needs real room to draw in. Writes go
    # exclusively through store.verify() (never a direct verified_* column
    # write from here), which is what keeps the invariant (non-empty
    # label, a real correction for 'verified_incorrect') airtight
    # regardless of how this flow branches in the UI.
    component = component_by_name.get(record.component_name)
    is_yolo = component is not None and component.model_type == "yolo"
    correcting_id = st.session_state.get("is_correcting_id")

    if record.verified_status == "unverified":
        # Collapsed by default and labeled "(optional)" — verifying is
        # never required to acknowledge/archive a record as usual; this
        # is purely an opt-in extra step for whoever wants to contribute
        # ground truth. Stays expanded mid-correction (correcting_id ==
        # this record) so an in-progress annotation is never hidden by
        # its own collapse.
        with st.expander("🔍 Verify this inspection (optional)", expanded=(correcting_id == record.id)):
            policy_note = {
                "off": "collected here, but this component's policy is currently 'off' — it won't be "
                "used for training until that's changed",
                "manual_review": "reviewed and selected before it's used for training",
                "automatic": "added directly to this component's training data",
            }.get(component.verified_correction_policy if component else None, "used for training")
            st.caption(
                "Optional — skip this and just Acknowledge/Archive as usual if you don't want to "
                "weigh in. If you do: tell the system what the correct answer actually was, a "
                "human-confirmed fact rather than the model's guess. Once verified, it's "
                + policy_note + ", helping the model get more accurate the next time it's retrained."
            )

            available_actions = ["correct"]
            if record.verdict == "failed":
                available_actions.append("false_positive")
            if record.verdict == "approved":
                available_actions.append("false_negative")
            if record.verdict == "failed" and is_yolo:
                available_actions.append("wrong_class")

            vcol1, vcol2 = st.columns([3, 1])
            action_key = vcol1.selectbox(
                "How does this compare to what actually happened?",
                available_actions,
                format_func=lambda k: VERIFY_ACTIONS[k]["label"],
                key=f"verify_action_{record.id}",
                label_visibility="collapsed",
            )
            st.caption(VERIFY_ACTIONS[action_key]["explain"])

            if vcol2.button("Apply", key=f"verify_apply_{record.id}"):
                if action_key == "correct":
                    # "Correct" means the prediction WAS the truth — the label
                    # is just a copy of the prediction, nothing to draw. Boxes
                    # are never part of that copy since a prediction's box
                    # coordinates were never persisted in the first place (see
                    # orchestrator.py's run_inspection() docstring); a
                    # verified_correct YOLO record with a 'failed' verdict
                    # simply won't have usable box data for retraining — see
                    # incorporate_verified_corrections()'s own handling of
                    # that case (skipped, stays pending, logged), not
                    # something this action should ask the operator to fix by
                    # annotating what it just confirmed was already right.
                    label = {"verdict": record.verdict, "defect_classes": record.defect_classes, "boxes": []}
                    store.verify(record.id, status="verified_correct", label=label, by=acknowledged_by or "operator")
                elif action_key == "false_positive":
                    label = {"verdict": "approved", "defect_classes": [], "boxes": []}
                    store.verify(record.id, status="verified_incorrect", label=label, by=acknowledged_by or "operator")
                elif action_key == "false_negative" and is_yolo:
                    st.session_state["is_correcting_id"] = record.id
                    st.session_state[f"flag_type_{record.id}"] = "false_negative"
                elif action_key == "false_negative":
                    label = {"verdict": "failed", "defect_classes": [], "boxes": []}
                    store.verify(record.id, status="verified_incorrect", label=label, by=acknowledged_by or "operator")
                elif action_key == "wrong_class":
                    st.session_state["is_correcting_id"] = record.id
                    st.session_state[f"flag_type_{record.id}"] = "wrong_class"
                st.rerun()

            if correcting_id == record.id:
                # Only false_negative/wrong_class ever reach here — "correct"
                # never does (see the Apply handler above: confirming a
                # prediction as correct always completes immediately, since
                # the label is just a copy of the prediction with nothing to
                # draw). Both remaining cases genuinely need a human-drawn
                # box because the truth differs from the prediction and no
                # correct box exists anywhere in the system to fall back on.
                flag_type = st.session_state.get(f"flag_type_{record.id}", "false_negative")
                if flag_type == "wrong_class":
                    action_verb = "Correcting the type — adjust the box/class below"
                else:
                    action_verb = "Correcting a missed defect — draw the correct box(es) and assign each a class"
                st.markdown(f"**Inspection {record.id}:** {action_verb}.")
                image_abs = paths.root / record.image_path if record.image_path else None
                if image_abs is None or not image_abs.exists():
                    st.warning("Image not available on disk — cannot annotate this correction.")
                else:
                    class_names = _yolo_classes_for(record.component_name)
                    pil_image = Image.open(image_abs)
                    initial_boxes = (
                        onboard.verified_label_boxes_to_yolo(record.verified_label, class_names)
                        if record.verified_label
                        else None
                    )
                    drawn_boxes = render_yolo_box_canvas(
                        pil_image, class_names, key_prefix=f"correct_{record.id}", initial_boxes=initial_boxes
                    )
                    ccol1, ccol2 = st.columns(2)
                    if ccol1.button("Save", key=f"save_correction_{record.id}", disabled=not drawn_boxes):
                        label = onboard.build_verified_label("failed", drawn_boxes, class_names)
                        store.verify(record.id, status="verified_incorrect", label=label, by=acknowledged_by or "operator")
                        st.session_state["is_correcting_id"] = None
                        st.session_state.pop(f"flag_type_{record.id}", None)
                        st.rerun()
                    if ccol2.button("Cancel", key=f"cancel_correction_{record.id}"):
                        st.session_state["is_correcting_id"] = None
                        st.session_state.pop(f"flag_type_{record.id}", None)
                        st.rerun()

    st.divider()
