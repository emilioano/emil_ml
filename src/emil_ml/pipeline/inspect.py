"""Single entry point for running inspection: raw input + component name -> result.

Used identically by the Streamlit UI and the folder watcher, so verdicts are
guaranteed consistent regardless of caller. Contains no Streamlit imports,
and no branching on modality or model_type — that dispatch lives entirely in
`core.registry_factory`. The raw input's shape (path, bytes, ...) and how
it's loaded/preprocessed is entirely up to the component's modality handler;
this function never inspects it directly.
"""

from __future__ import annotations

from typing import Any

from emil_ml.config.registry import ComponentRegistry
from emil_ml.core import registry_factory
from emil_ml.core.base import PredictionResult
from emil_ml.core.reporting import reporter


def inspect(
    raw_input: Any,
    component_name: str,
    *,
    threshold_override: float | None = None,
    registry: ComponentRegistry | None = None,
    include_report: bool = True,
) -> dict:
    """Run inspection for `raw_input` against the named component's model.

    `raw_input` is whatever the component's modality expects as loadable input
    (for the "image" modality: a file path, raw bytes, a PIL.Image, or an
    (H, W, 3) numpy array) — this function delegates loading/preprocessing to
    the modality handler and never assumes a shape itself.

    `threshold_override` lets a caller (e.g. a session-only UI setting) re-derive
    the verdict at a different threshold without touching the component's stored
    default. It has no effect on methods that don't use a threshold (their
    result's `threshold` is None).

    `include_report=False` skips report generation even if the component's
    reporting settings would otherwise call for one — for a caller that
    wants the fast verdict now and plans to generate the (possibly slow,
    LLM-backed) report itself afterward, e.g. asynchronously. See
    core/inspections/orchestrator.py, which is the caller that actually
    does this — detection and persistence of the fast verdict must never
    block on report generation.

    Returns {"verdict": "approved"|"failed", "score": float, "threshold": float|None,
             "component": str, "details": dict}, plus "report" (a
             core.reporting.reporter.ReportResult) when a report was
             generated inline.
    """
    registry = registry or ComponentRegistry()
    component = registry.get(component_name)
    if component is None:
        raise KeyError(f"No component named {component_name!r}")
    if component.status != "ready":
        raise ValueError(
            f"Component {component_name!r} is not ready for inspection (status={component.status!r})"
        )

    handler = registry_factory.get_modality_handler(component.modality, component)
    prepared_input = handler.load(raw_input)

    predictor = registry_factory.get_predictor(component.modality, component.model_type, component)
    result = predictor.predict(prepared_input)

    if threshold_override is not None and result.threshold is not None:
        verdict = "approved" if result.score <= threshold_override else "failed"
        result = PredictionResult(
            verdict=verdict, score=result.score, threshold=threshold_override, details=result.details
        )

    output = {
        "verdict": result.verdict,
        "score": result.score,
        "threshold": result.threshold,
        "component": component.name,
        "details": result.details,
    }

    # The only point in the pipeline that knows about core.reporting at
    # all. When should_generate_report() returns False (reporting_enabled
    # off, or the configured condition doesn't apply to this verdict),
    # nothing below runs and `output` is byte-for-byte what it was before
    # RAG existed.
    if include_report and reporter.should_generate_report(component, result):
        output["report"] = reporter.generate_report(component, result)

    return output
