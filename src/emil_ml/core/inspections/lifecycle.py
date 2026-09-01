"""File-side lifecycle for inspections: saving the analyzed image, writing
the human-readable .report.md beside it, and moving both through
input/ -> analyzed/{approved|failed}/ -> archive/ as an operator
acknowledges and archives an inspection.

The database (store.py) is authoritative — everything here is an
artifact of what's already recorded there, not the other way around. A
.report.md can always be regenerated from a stored InspectionRecord.

Write ordering matters and is deliberate throughout this module: a file
is always written/moved to its destination BEFORE the database row that
points at it is created or updated. If the file operation fails, no DB
row ends up pointing at a file that doesn't exist; if the DB write fails
after a successful file operation, the result is an orphaned-but-readable
file — the lesser problem, and one a human can find just by looking in
the folder.

No automatic deletion lives here. Component.inspection_retention_days
(config/registry.py) is a real, stored setting, but nothing reads it yet
— a future cleanup job is what would use it to decide which archived
files are old enough to remove. Acknowledging an inspection only flips
its DB status (see store.acknowledge()); archiving only moves files
already-acknowledged inspections point at. Nothing in this project ever
deletes an inspection's files as a side effect of a button click.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from emil_ml.core.inspections import store
from emil_ml.core.inspections.store import InspectionRecord
from emil_ml.core.reporting.reporter import ReportResult
from emil_ml.utils import image_io
from emil_ml.utils.image_io import ImageInput
from emil_ml.utils.paths import ComponentPaths, for_component

logger = logging.getLogger(__name__)

_TIMESTAMP_FMT = "%Y%m%d_%H%M%S_%f"


def save_analyzed_image(paths: ComponentPaths, raw_input: ImageInput, verdict: str) -> Path:
    """Write the inspected image into analyzed/{approved|failed}/, returning its absolute path.

    Named with a microsecond timestamp, not the camera's/upload's
    original filename: a watcher pulling repeatedly-numbered frames
    (e.g. "017.png" on every cycle) would otherwise silently overwrite
    the previous "017.png" sitting in analyzed/ before it's ever
    acknowledged. archive() applies its own, separate uniqueness
    guarantee (timestamp + inspection id) for the same reason at the
    next stage of the lifecycle.
    """
    target_dir = paths.analyzed_approved_dir if verdict == "approved" else paths.analyzed_failed_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{datetime.now(timezone.utc).strftime(_TIMESTAMP_FMT)}.png"
    target_path = target_dir / filename
    image_io.save_image(raw_input, target_path)
    return target_path


def _report_markdown_text(
    *, component_name: str, verdict: str, score: float, created_at: str, report: ReportResult | None
) -> str:
    frontmatter = [
        "---",
        f"component: {component_name}",
        f"verdict: {verdict}",
        f"score: {score}",
        f"timestamp: {created_at}",
        "---",
        "",
    ]
    if report is None:
        body = ["No report was generated for this inspection."]
    else:
        body = [report.report_text, ""]
        if report.machine_context_used:
            body.append("## Machine context considered")
            for state in report.machine_context_used:
                body.append(f"- {state}")
            body.append("")
        if report.sources:
            body.append("## Sources")
            for s in report.sources:
                body.append(f"- {s['source']} — {s['section']} ({s['doc_type']}) — `{s['path']}`")
            body.append("")
        if report.thinking_used:
            body.append("## Model reasoning (raw)")
            body.append(report.thinking_used)
            body.append("")
        if report.prompt_used:
            body.append("## Prompt sent to the LLM")
            body.append("```")
            body.append(report.prompt_used)
            body.append("```")
    return "\n".join(frontmatter + body) + "\n"


def write_report_markdown(
    image_path: Path,
    *,
    component_name: str,
    verdict: str,
    score: float,
    created_at: str,
    report: ReportResult | None,
) -> Path:
    """<image stem>.report.md, written next to `image_path` (017.png -> 017.report.md).

    Self-explanatory for someone who just opens the folder: YAML
    frontmatter (verdict/score/timestamp/component), then the report
    text, then machine context and sources in readable form. Written
    even for the "no documentation found" case (Fas 4) — an honest
    report is still a report, and still gets persisted and shown.
    """
    report_path = image_path.with_name(f"{image_path.stem}.report.md")
    text = _report_markdown_text(
        component_name=component_name, verdict=verdict, score=score, created_at=created_at, report=report
    )
    report_path.write_text(text, encoding="utf-8")
    return report_path


def archive(paths: ComponentPaths, record: InspectionRecord) -> tuple[str | None, str | None]:
    """Move an acknowledged inspection's image (+ .report.md, if any) into
    today's date-partitioned archive folder, with a name unique enough
    that the camera's own recurring filenames can never collide here.

    Returns (new_image_path, new_report_path), both relative to the
    component root — the caller (orchestrator.py) passes these straight
    to store.mark_archived() in the same operation, so the DB row is
    never left pointing at the pre-move location.
    """
    if record.status != "acknowledged":
        raise ValueError(
            f"Inspection {record.id} must be 'acknowledged' before archiving (is {record.status!r}) — "
            "archiving is a separate, deliberate step from acknowledgement, not automatic."
        )

    today = datetime.now(timezone.utc).date()
    dest_dir = paths.archive_dir_for_date(today)
    dest_dir.mkdir(parents=True, exist_ok=True)
    unique_stem = f"{datetime.now(timezone.utc).strftime(_TIMESTAMP_FMT)}_{record.id}"

    new_image_path: str | None = None
    if record.image_path:
        source_image = paths.root / record.image_path
        if source_image.exists():
            dest_image = dest_dir / f"{unique_stem}{source_image.suffix}"
            shutil.move(str(source_image), str(dest_image))
            new_image_path = dest_image.relative_to(paths.root).as_posix()

    new_report_path: str | None = None
    if record.report_path:
        source_report = paths.root / record.report_path
        if source_report.exists():
            dest_report = dest_dir / f"{unique_stem}.report.md"
            shutil.move(str(source_report), str(dest_report))
            new_report_path = dest_report.relative_to(paths.root).as_posix()

    logger.info(
        "component=%s archived inspection id=%s: image %s -> %s, report %s -> %s",
        record.component_name, record.id, record.image_path, new_image_path, record.report_path, new_report_path,
    )
    return new_image_path, new_report_path


@dataclass(frozen=True)
class BulkArchiveResult:
    """Outcome of one bulk_archive_approved() call — a report, not just a
    count, so the caller (the Inspection Station's confirmation dialog)
    can show the hard guard actually happened rather than silently
    dropping ineligible records (see that function's own docstring)."""

    archived: int
    skipped_failed_verdict: int
    skipped_already_archived: int
    errors: list[tuple[int, str]] = field(default_factory=list)


def bulk_archive_approved(
    inspection_ids: list[int],
    *,
    by: str,
    on_progress: Callable[[int, int], None] | None = None,
) -> BulkArchiveResult:
    """Archive every given inspection that is verdict='approved' — the
    bulk counterpart to the Inspection Station's single-record Archive
    button, built for the "thousands of approved records" case a
    production line generates. Acknowledges first if a record is still
    'new' (the single-record button never needs to, since it only ever
    appears once a record is already acknowledged; a bulk selection can
    mix 'new' and 'acknowledged' approved records in one batch).

    Hard, non-negotiable guard: a 'failed' verdict is NEVER archived by
    this function, regardless of how it ended up in `inspection_ids` —
    the caller's UI should avoid offering failed records for a bulk
    archive selection in the first place (see the Inspection Station's
    confirmation dialog), but this is the actual enforcement point, not
    just a UI nicety. Skipped and counted in the returned result, never
    silently dropped without being reported back.

    Resumable by construction, not by any special-cased "resume" logic:
    each record's archive is its own atomic file-move + DB update
    (reusing archive()/store.mark_archived() exactly as the single-record
    action does), and any inspection already `status == 'archived'` is
    skipped immediately — the same status-field-as-source-of-truth
    principle the folder watcher already relies on to resume cleanly
    after a restart (see watcher/service.py). Re-running this function
    with the exact same `inspection_ids` after an interruption (a closed
    browser tab, a killed process) is always safe: already-archived
    records are skipped, nothing is double-processed.

    One bad record (a missing file, a stale id) is logged into `errors`
    and does not abort the rest of the batch — a single failure among
    thousands shouldn't block every other legitimate archive.

    `on_progress(done, total)` fires after every record, purely for a
    caller that wants to render a progress bar — same on_progress
    callback pattern core/reporting/reporter.py already uses for the
    Inspect page's live report-generation view.
    """
    archived = 0
    skipped_failed = 0
    skipped_already = 0
    errors: list[tuple[int, str]] = []
    total = len(inspection_ids)

    for i, inspection_id in enumerate(inspection_ids, start=1):
        try:
            record = store.get(inspection_id)
            if record is None:
                errors.append((inspection_id, "inspection no longer exists"))
                continue
            if record.status == "archived":
                skipped_already += 1
                continue
            if record.verdict != "approved":
                skipped_failed += 1
                continue
            if record.status == "new":
                store.acknowledge(inspection_id, by=by)
                record = store.get(inspection_id)
                assert record is not None

            paths = for_component(record.component_name)
            new_image_path, new_report_path = archive(paths, record)
            store.mark_archived(inspection_id, image_path=new_image_path, report_path=new_report_path)
            archived += 1
        except Exception as exc:  # noqa: BLE001 - one bad record must not abort a large batch
            logger.exception("bulk archive: inspection id=%s failed", inspection_id)
            errors.append((inspection_id, str(exc)))
        finally:
            if on_progress:
                on_progress(i, total)

    logger.info(
        "bulk archive complete: archived=%d skipped_failed_verdict=%d skipped_already_archived=%d errors=%d",
        archived, skipped_failed, skipped_already, len(errors),
    )
    return BulkArchiveResult(
        archived=archived,
        skipped_failed_verdict=skipped_failed,
        skipped_already_archived=skipped_already,
        errors=errors,
    )
