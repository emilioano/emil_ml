"""Orchestrates core/reporting into a ReportResult for one inspection.

Two things live here:

- `should_generate_report()` — the single gating decision point.
  pipeline.inspect() calls this and nothing else in core.reporting when
  it returns False, so a component with reporting_enabled off never
  touches retrieval, machine context, or the LLM — behavior identical to
  before RAG existed. Unchanged since Fas 3.
- `generate_report()` — chains machine context (Fas 3) -> retrieval
  (Fas 2) -> prompt building (prompt.py) -> generation (llm.py) into a
  ReportResult. Two branches, both final:
    - No relevant documentation retrieved: an honest "nothing found"
      report_text, no LLM call at all. This branch predates Fas 4 and is
      NOT changed by it — the LLM is never given a chance to invent
      content when there's nothing to ground it in.
    - Documentation found: builds a prompt and calls the configured LLM
      backend (mock by default — see config/settings.py's
      DEFAULT_LLM_MODE) for the actual narrative.
  Either way, a component can have reporting on with no knowledge-base
  documents and no machine-parameter definitions at all; generate_report()
  still returns a meaningful ReportResult, not an all-or-nothing failure.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Callable

from emil_ml.config.registry import Component
from emil_ml.config.settings import DEFAULT_LLM_MODE
from emil_ml.core.base import PredictionResult
from emil_ml.core.reporting import llm
from emil_ml.core.reporting import prompt as prompt_builder
from emil_ml.core.reporting.knowledge import retriever
from emil_ml.core.reporting.knowledge.retriever import RetrievedChunk
from emil_ml.core.reporting.machine_context import analyzer
from emil_ml.core.reporting.machine_context.source import MachineContextSource, SqliteMachineContextSource
from emil_ml.utils.paths import for_component

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class ReportResult:
    """Structured report output — matches the shape reporter.py was designed
    around from the start: text, sources, machine context, and (once Fas 4
    exists) the prompt/model used, kept for debugging."""

    report_text: str
    sources: list[dict[str, str]]  # [{"source": ..., "section": ..., "doc_type": ..., "path": ...}, ...]
    machine_context_used: list[str]  # searchable states folded into the retrieval query
    prompt_used: str | None  # None until Fas 4 builds an actual prompt
    model: str | None  # None until Fas 4 calls an actual LLM
    thinking_used: str | None = None  # a "thinking" model's chain-of-thought, if any (debugging only)
    error: str | None = None  # set when llm.generate() degraded (see llm.LLMResult.error) — report_text
    # is still a real, honest message in that case (never blank), but the caller (orchestrator.py)
    # needs this to persist report_status='failed' instead of 'complete' — without it, a degraded
    # generation (Ollama unreachable, empty response, ...) was previously indistinguishable from a
    # genuinely successful report anywhere past this function, including the Inspection Station's
    # own report_status badge (confirmed happening in practice: it showed "Report available" for a
    # report whose actual body was an apology that Ollama couldn't be reached).


def _cited_chunks(report_text: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Which of the retrieved `chunks` the model actually cited in `report_text`.

    prompt.py's "Reference documentation" list is 1-indexed in prompt
    order (chunk[0] is [1], chunk[1] is [2], ...) — the same numbering
    the model was instructed to cite by by. Retrieving a chunk only means
    it was plausibly relevant enough to show the model, not that the
    report actually leans on it; a report's Sources section should
    reflect what it was actually built from. Out-of-range citations (a
    number beyond what was ever offered) are ignored rather than raising
    — the model misciting is a generation-quality issue, not a reason to
    crash report assembly. Order follows the original retrieval order,
    not citation order in the text.

    Scans the WHOLE report_text, including the model's own "## Sources"
    section — not just the body. Tried restricting this to the body only
    (on the theory that only an inline citation really means "drew on
    this"), but that broke a real, common case: a model that correctly
    lists its real sources in "## Sources" without also repeating an
    inline [n] marker in the prose (prompt.py asks for both, but
    inline-citation compliance is the less reliable half) — restricting
    to the body then wrongly produced an EMPTY ReportResult.sources for
    a report that was genuinely, correctly grounded. The actual failure
    mode this function exists to prevent (padding "## Sources" with
    numbers never drawn on anywhere) is prompt.py's job to stop at the
    source — see its explicit "an empty Sources section is the correct,
    honest output" instruction — not something to work around here by
    distrusting the model's own citation section.
    """
    cited_indices = {int(n) for n in _CITATION_RE.findall(report_text)}
    return [chunk for i, chunk in enumerate(chunks, start=1) if i in cited_indices]


def _source_entry(component_name: str, chunk: RetrievedChunk) -> dict[str, str]:
    """One Sources-list entry, including the chunk's relative path — component-root-relative,
    the same convention InspectionRecord.image_path/report_path and Component.model_path
    already use (see utils/paths.py's ComponentPaths), derived from existing path
    properties rather than reconstructed by hand."""
    paths = for_component(component_name)
    path = (paths.knowledge_dir / chunk.file).relative_to(paths.root).as_posix()
    return {"source": chunk.source, "section": chunk.section, "doc_type": chunk.doc_type, "path": path}


def _extract_defect_class(prediction: PredictionResult) -> str | None:
    """Best-effort defect-class extraction from PredictionResult.details.

    Genuinely method-dependent and not yet unified across detectors:
    YOLO's details has "detections" (a list of boxes, each with its own
    "class"); the classifier's is a binary failed/approved probability
    with no named class; autoencoder/PatchCore/Isolation Forest never
    have one at all (that's exactly the "no defect label" case machine
    context matters most for). Returns None for anything unrecognized —
    Fas 4's prompt-building is the natural place to generalize this
    further if needed.
    """
    detections = prediction.details.get("detections")
    if detections:
        return detections[0].get("class")
    return prediction.details.get("predicted_class")


def should_generate_report(component: Component, prediction: PredictionResult) -> bool:
    """The single point that decides whether a report gets generated for this inspection.

    pipeline.inspect() calls this once, after prediction, and calls
    nothing else in core.reporting when it returns False.
    """
    if not component.reporting_enabled:
        return False

    condition = component.reporting_condition
    if condition == "never":
        return False
    if condition == "always":
        return True
    if condition == "on_failed":
        return prediction.verdict == "failed"
    if condition == "on_classes":
        classes = json.loads(component.reporting_classes or "[]")
        return _extract_defect_class(prediction) in classes
    raise ValueError(f"Unknown reporting_condition {component.reporting_condition!r}")


def generate_report(
    component: Component,
    prediction: PredictionResult,
    *,
    machine_context_source: MachineContextSource | None = None,
    llm_mode: str = DEFAULT_LLM_MODE,
    on_progress: Callable[[str], None] | None = None,
    on_chunk: Callable[[str, str], None] | None = None,
) -> ReportResult:
    """Assemble a report from whatever's actually configured/available.

    Never all-or-nothing: a component with no machine_parameters defined
    simply contributes no machine context (analyzer.analyze() already
    handles that — see its own docstring); a component with no indexed
    knowledge-base documents gets an honest "nothing found" report_text
    and never reaches the LLM at all (see retriever.retrieve()'s own
    empty-result handling, Fas 2) — that branch is unchanged by Fas 4.

    `on_progress(message)` fires at each pipeline stage, `on_chunk(kind,
    text_so_far)` passes straight through to llm.generate() (see its own
    docstring) — both purely for a caller that wants to show live
    progress (orchestrator.py wires these to core/inspections/progress.py
    for app/pages/1_inspect.py's auto-refreshing report section). Neither
    changes what's returned; generate_report() behaves identically with
    or without them.
    """

    def _progress(message: str) -> None:
        if on_progress:
            on_progress(message)

    machine_context_source = machine_context_source or SqliteMachineContextSource()
    _progress("Reading machine context...")
    reading = machine_context_source.get_readings(component.name)
    machine_ctx = analyzer.analyze(reading, component)
    if machine_ctx.has_anomalies:
        logger.debug("component=%s machine context anomalies: %s", component.name, machine_ctx.searchable_states)
        _progress(f"Machine context: {', '.join(machine_ctx.searchable_states)}.")
    else:
        logger.debug("component=%s machine context: no anomalies (reading=%s)", component.name, reading)
        _progress("Machine context: no anomalies.")

    defect_class = _extract_defect_class(prediction)
    _progress("Retrieving relevant documentation...")
    chunks = retriever.retrieve_for_inspection(
        component.name, defect_class=defect_class, machine_states=machine_ctx.searchable_states
    )

    if not chunks:
        logger.warning(
            "component=%s no relevant documentation found (defect_class=%s, machine_states=%s)",
            component.name, defect_class, machine_ctx.searchable_states,
        )
        _progress("No relevant documentation found.")
        return ReportResult(
            report_text="No relevant documentation was found in the knowledge base for this component.",
            sources=[],
            machine_context_used=machine_ctx.searchable_states,
            prompt_used=None,
            model=None,
        )

    logger.debug("component=%s %d chunk(s) retrieved, building prompt (llm_mode=%s)", component.name, len(chunks), llm_mode)
    _progress(f"Found {len(chunks)} relevant document chunk(s). Building prompt...")
    prompt = prompt_builder.build_prompt(
        component.name, prediction, machine_ctx, chunks, defect_class=defect_class
    )
    _progress("Sending prompt to the LLM...")
    llm_result = llm.generate(prompt, mode=llm_mode, on_chunk=on_chunk)
    if llm_result.error:
        logger.warning("component=%s LLM generation degraded: %s", component.name, llm_result.error)
    _progress("Done.")

    # Sources reflect what the report actually cites ([n] markers in the
    # generated text), not everything that was merely retrieved — see
    # _cited_chunks()'s own docstring for why that distinction matters.
    sources = [_source_entry(component.name, c) for c in _cited_chunks(llm_result.text, chunks)]

    return ReportResult(
        report_text=llm_result.text,
        sources=sources,
        machine_context_used=machine_ctx.searchable_states,
        prompt_used=prompt,
        model=llm_result.model,
        thinking_used=llm_result.thinking,
        error=llm_result.error,
    )
