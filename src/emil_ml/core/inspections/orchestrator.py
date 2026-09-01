"""Ties pipeline.inspect() together with persistence (store.py) and file
lifecycle (lifecycle.py) — what the UI (and, eventually, a folder
watcher) calls instead of pipeline.inspect() + reporter.generate_report()
directly.

`run_inspection()` is the one place "run detection, persist immediately,
generate the report afterward without blocking on it" is orchestrated:

1. Run detection (pipeline.inspect(..., include_report=False) — fast,
   synchronous).
2. Save the image into analyzed/{approved|failed}/ and create the
   `inspections` row immediately — the verdict is already final and
   worth having durably recorded even if report generation is slow, or
   never happens at all.
3. If (and only if) this component's reporting settings call for one
   (reporter.should_generate_report()), queue the report onto
   report_worker's single, process-wide serialized worker and update the
   same row when it finishes — report_status starts 'pending' and the
   caller can poll store.get() to see it move to 'complete' or 'failed'.
   If reporting doesn't apply, report_status is 'none' from the start
   and nothing else happens.

Detection and the durable "new" record never wait on the LLM — only the
report fields do. Report generation itself is serialized against every
other report in this process (see report_worker.py) — never spawned as
its own concurrent thread per call — since Ollama can only run one
qwen3:8b generation at a time on a single GPU with reasonable throughput;
several running "at once" doesn't parallelize, it just makes each one
slow enough to blow through its own timeout waiting behind the others.
"""

from __future__ import annotations

import logging

from emil_ml.config.registry import Component, ComponentRegistry
from emil_ml.core.base import PredictionResult
from emil_ml.core.inspections import lifecycle, progress, report_worker, store
from emil_ml.core.reporting import reporter
from emil_ml.pipeline.inspect import inspect
from emil_ml.utils.image_io import ImageInput
from emil_ml.utils.paths import for_component

logger = logging.getLogger(__name__)


def _extract_defect_classes(details: dict) -> list[str]:
    """All distinct defect classes visible in a prediction's details, if any.

    Same method-dependent extraction reporter.py's `_extract_defect_class`
    does, but plural — YOLO can report more than one class across its
    detections in a single image, so the inspections table's
    defect_classes column is a list, not a single value.
    """
    detections = details.get("detections")
    if detections:
        return sorted({d["class"] for d in detections if d.get("class")})
    predicted = details.get("predicted_class")
    return [predicted] if predicted else []


def _generate_report_and_update(component: Component, prediction: PredictionResult, inspection_id: int) -> None:
    paths = for_component(component.name)
    progress.start(inspection_id)
    logger.info("component=%s report generation started (inspection id=%s)", component.name, inspection_id)
    try:
        report = reporter.generate_report(
            component,
            prediction,
            on_progress=lambda message: progress.add_stage(inspection_id, message),
            on_chunk=lambda kind, text: progress.set_chunk(inspection_id, **{kind: text}),
        )
        record = store.get(inspection_id)
        assert record is not None
        report_path = lifecycle.write_report_markdown(
            paths.root / record.image_path,
            component_name=component.name,
            verdict=record.verdict,
            score=record.score,
            created_at=record.created_at,
            report=report,
        )
        # llm.generate() degrades honestly instead of raising when Ollama
        # can't be reached or returns nothing usable (see LLMResult.error) —
        # report.report_text is still a real, human-readable message in
        # that case, never blank, so it's still worth persisting and
        # writing to disk above. But it must land as report_status='failed'
        # here, not 'complete' — otherwise the Inspection Station's badge
        # (and everything else keying off report_status) can't tell a
        # genuine report apart from an apology that generation didn't work.
        report_status = "failed" if report.error else "complete"
        store.update_report(
            inspection_id,
            report_text=report.report_text,
            report_sources=report.sources,
            machine_context_used=report.machine_context_used,
            report_model=report.model,
            report_path=report_path.relative_to(paths.root).as_posix(),
            report_status=report_status,
            report_prompt=report.prompt_used,
            report_thinking=report.thinking_used,
        )
        if report.error:
            logger.warning(
                "component=%s report generation degraded (inspection id=%s): %s",
                component.name, inspection_id, report.error,
            )
        else:
            logger.info(
                "component=%s report generation complete (inspection id=%s, model=%s, %d source(s))",
                component.name, inspection_id, report.model, len(report.sources),
            )
    except Exception as exc:  # noqa: BLE001 - a background thread; must not crash silently or hang the row on 'pending'
        # This is the one place a background-thread failure could
        # otherwise vanish completely — store.mark_report_failed() below
        # records it in the DB (so the UI shows it), but nothing else
        # would ever surface it without this log line. Full traceback via
        # .exception(), not just str(exc): an unexpected failure here
        # (as opposed to Ollama's own "couldn't reach it" degrade-honest
        # path, which already returns a normal LLMResult) is exactly the
        # kind of thing worth being able to actually debug afterward.
        logger.exception("component=%s report generation FAILED (inspection id=%s): %s", component.name, inspection_id, exc)
        store.mark_report_failed(inspection_id, str(exc))
    finally:
        progress.clear(inspection_id)


def run_inspection(
    raw_input: ImageInput,
    component_name: str,
    *,
    threshold_override: float | None = None,
    registry: ComponentRegistry | None = None,
    async_report: bool = True,
    run_by: str | None = None,
) -> tuple[store.InspectionRecord, dict]:
    """Run detection, persist immediately, generate the report (if this
    component's settings call for one) in the background by default.

    `async_report=False` runs report generation inline before returning
    instead — useful for scripts/tests that want a fully-populated
    InspectionRecord back without needing to poll.

    `run_by` records who/what ran this inspection (see store.create()'s
    own docstring) — the Streamlit Inspect page passes an operator name,
    the folder watcher passes "watcher". Distinct from acknowledgement,
    which is always a separate, later, human action.

    Returns `(record, details)` — `details` is the predictor's raw,
    method-specific `PredictionResult.details` (e.g. YOLO's `detections`,
    PatchCore's `heatmap` array) for a caller that wants to render it
    immediately (see app/pages/1_inspect.py). It's deliberately NOT part
    of the persisted InspectionRecord/`inspections` table — a heatmap is
    a numpy array, not JSON-shaped data, and the table's defect_classes
    column already captures the one part of `details` worth querying
    later.
    """
    registry = registry or ComponentRegistry()
    component = registry.get(component_name)
    if component is None:
        raise KeyError(f"No component named {component_name!r}")

    result = inspect(
        raw_input, component_name, threshold_override=threshold_override, registry=registry, include_report=False
    )
    prediction = PredictionResult(
        verdict=result["verdict"], score=result["score"], threshold=result["threshold"], details=result["details"]
    )

    paths = for_component(component_name)
    image_path_abs = lifecycle.save_analyzed_image(paths, raw_input, result["verdict"])

    will_report = reporter.should_generate_report(component, prediction)
    record = store.create(
        component_name,
        verdict=result["verdict"],
        score=result["score"],
        threshold=result["threshold"],
        defect_classes=_extract_defect_classes(result["details"]),
        image_path=image_path_abs.relative_to(paths.root).as_posix(),
        report_status="pending" if will_report else "none",
        run_by=run_by,
    )
    logger.info(
        "component=%s inspection complete: verdict=%s score=%.6f moved_to=%s (id=%s)",
        component_name, record.verdict, record.score, record.image_path, record.id,
    )

    # Approved units on a component set to auto_acknowledge skip the human
    # acknowledgement step entirely — see settings.py's
    # VALID_APPROVED_HANDLING_MODES for why this is safe: acknowledging
    # only flips a DB status (files untouched, nothing archived yet), so
    # the record is still fully reachable and still goes through the
    # normal retention-based archiving later, not immediately. A "by"
    # value that's obviously not a person, unlike acknowledge()'s usual
    # human-entered names, so the Inspection Station can show at a glance
    # this wasn't a human decision.
    if record.verdict == "approved" and component.approved_handling == "auto_acknowledge":
        store.acknowledge(record.id, by="system (auto_acknowledge)")
        record = store.get(record.id)
        assert record is not None
        logger.info("component=%s inspection id=%s auto-acknowledged (approved_handling=auto_acknowledge)", component_name, record.id)

    if will_report:
        if async_report:
            # Queued, not spawned as its own thread — see report_worker.py's
            # module docstring for why: several reports' Ollama calls
            # running concurrently doesn't parallelize, it just makes
            # each one slow enough to time out waiting behind the others
            # for the same GPU. The single worker drains this queue one
            # report at a time; this call still returns immediately.
            report_worker.enqueue(lambda: _generate_report_and_update(component, prediction, record.id))
        else:
            _generate_report_and_update(component, prediction, record.id)
            record = store.get(record.id)
            assert record is not None

    return record, result["details"]
