"""Builds the LLM prompt from three inputs: the detection result, machine
context (Fas 3), and retrieved knowledge-base chunks (Fas 2).

Pure string assembly — no I/O, no model calls (see llm.py for that) — so
it's independently testable and inspectable: ReportResult.prompt_used is
exactly this function's return value, which is the whole point of keeping
prompt-building split out from generation (see reporter.py's docstring).
Deterministic: the same three inputs always produce the exact same
string, so a bad report is easy to debug by reading the prompt back.

This is the diagnostic core of the reporting layer. The instructions
below exist specifically to prevent an LLM from doing the thing they do
by default — sounding confident about things it wasn't actually given:

- Gate on Verdict FIRST, before any defect/anomaly reasoning. An
  approved verdict means there is nothing to diagnose, full stop —
  every other rule (classified defect, unclassified anomaly, machine
  context as a cause) only applies once Verdict is "failed". Added
  after a real failure: a component with reporting_condition="always"
  generated a full defect report — quarantine recommendation included —
  for an APPROVED unit with a 0.0 anomaly score, because "Detected
  defect class: none" was being read the same way regardless of
  verdict. "none" is genuinely ambiguous on its own — on a failed
  verdict it means "an anomaly was found but not classified"; on an
  approved verdict it just means "no defects, because there's nothing
  to classify" — and the rules previously only spelled out the first
  reading. Gating on verdict before anything else removes that
  ambiguity structurally instead of hoping the model infers it.
- Ground ONLY in the provided documentation excerpts, and only reason
  about a connection that's actually indicated for this inspection (a
  detected defect class or a measured anomaly) — not about every topic
  that happens to be in the retrieved documentation just because it's
  in the prompt. No general maintenance advice invented from the
  model's own training data.
- State a hypothesis, never a settled cause. Correlating a visual
  anomaly with machine parameters and documentation text is not proof of
  causality.
- When there's no defect class (PatchCore/autoencoder give an anomaly
  with no label), lean on machine-context anomalies instead of guessing
  a defect type — and if there's neither a defect class nor a machine
  anomaly, say so honestly rather than inventing a lead.
- When machine context reports no anomalies (parameters checked and
  found normal, or no data at all), that's evidence AGAINST machine
  settings as a cause, not silence to fill with speculation — lean on
  the defect's own documented non-machine causes instead. This is the
  rule that stops the self-contradicting pattern seen in practice: a
  report stating "no machine-context anomalies were recorded" and then
  speculating about machine parameters as a possible cause in the next
  breath anyway.
- The [n] citation markers belong ONLY to the numbered "Reference
  documentation" list — ReportResult.sources is built exclusively from
  those retrieved chunks (see reporter.py), machine context was never
  part of that numbering. Confirmed happening in practice: an earlier
  version of this prompt told the model to cite "wherever you draw on
  it" without scoping that to documentation, and the model sometimes
  tried to cite its own machine-context reasoning too — with no real
  number reserved for it, it would write the literal text "[n]" into
  the report instead of an actual citation. The rule now explicitly
  scopes [n] to documentation and tells the model to describe machine
  context in plain language instead.
- The model's own "## Sources" section must spell out what each number
  means (title + section, copied from that excerpt's own "Source: ...
  — Section: ..." line), not just list bare numbers — a reader
  shouldn't have to cross-reference anything to know what a [2] inline
  refers to. It must also stay consistent with the inline citations:
  every [n] cited in the body appears in Sources, and Sources contains
  nothing that wasn't actually cited (confirmed happening in practice:
  the model citing [3] inline but leaving it out of its own Sources
  line). This is deliberately in addition to, not instead of, the
  structured Sources list app/pages/1_inspect.py and 3_history.py
  render from ReportResult.sources (see reporter.py's _cited_chunks())
  — that one carries the relative file path for traceability; this one
  lets a reader interpret an inline [n] in place, in the model's own
  prose.
"""

from __future__ import annotations

from emil_ml.core.base import PredictionResult
from emil_ml.core.reporting.knowledge.retriever import RetrievedChunk
from emil_ml.core.reporting.machine_context.analyzer import MachineContext

_INSTRUCTIONS = """You are a manufacturing quality assistant. You write short, grounded maintenance notes from an automated visual inspection result, using ONLY the reference documentation provided below.

Rules:
1. Check "Verdict" in the Detection result below FIRST, before anything else — it decides which of the rules below even apply.
   If Verdict is "approved": the unit PASSED inspection. There is NO defect and NO anomaly to explain, regardless of what "Detected defect class" says or what documentation happens to be attached below. Your report must be a short, factual confirmation: the inspection passed, the anomaly score was under threshold, no defects were detected, no action is needed on this unit. Do NOT recommend quarantine, reason about defect mechanisms, speculate about causes, treat retrieved defect documentation as a "possible explanation" for anything, or suggest inspecting stations — there is nothing to diagnose. Documentation being provided below does not mean you have to use it: if you're not citing anything, write "## Sources" followed by nothing — an empty Sources section is the CORRECT, honest output when nothing was actually drawn on, not a section to fill just because it exists. If the "Machine context" section reports an anomaly even though the unit passed (e.g. overdue service on an otherwise-good unit), you may mention it as a preventive process note (e.g. "service is overdue and worth scheduling, though this unit itself passed inspection") — never as a defect on this unit, never with a quarantine or corrective-action recommendation. Still use the four-section structure from rule 6, just filled with approved-appropriate content — none of rules 2-5 below apply to an approved verdict.
   Only if Verdict is "failed" do rules 2-5 apply — everything below this point assumes a failed verdict.
2. Base your reasoning ONLY on the documentation excerpts under "Reference documentation" below. Do not add general maintenance advice that sounds authoritative but isn't grounded in those excerpts. If the documentation doesn't cover something relevant to this case, say so explicitly rather than inventing an answer. Only reason about a causal connection — to a machine parameter, a defect mechanism, anything — if it is supported by an actual indication for THIS inspection (the detected defect class, or a measured machine-parameter anomaly). A topic appearing in the reference documentation below is not itself an indication; documentation is retrieved because it's plausibly relevant, not because every excerpt in it applies to this specific case.
3. State a HYPOTHESIS, never a settled cause. This system correlates a visual anomaly with machine-parameter values and documentation text — that is not proof of causality. Use phrasing like "likely cause" or "worth investigating first", never "this is the defect" or "this is what caused it".
4. If the "Machine context" section below reports that all monitored parameters were within their normal range (or that no machine data was available), do NOT speculate that machine parameters were a contributing cause anyway. Parameters being normal is itself informative — it points AWAY from machine settings as a cause, not toward them. State that plainly, then base your hypothesis on the defect's own documented, non-machine causes from the reference documentation (material, component, handling, process — whatever it actually attributes the defect to). Do not hedge this into "no anomalies were found, but perhaps the parameters were still involved" — that contradicts the data you were just given.
5. If "Detected defect class" below is "none", the detector found an anomaly without identifying what kind — a REAL unclassified anomaly: the score was over threshold (Verdict is "failed", per rule 1) but the detector couldn't say what type of defect it is. Do not confuse this with an approved unit, where "none" just means "no defects" and rule 1 already applies instead. In this genuinely-unclassified-failed case: if the "Machine context" section reports an actual anomaly, lean your reasoning on it — that is the strongest available lead here. If it does NOT (no defect class AND no machine anomaly), say so honestly: an anomaly was detected but not classified, and no machine parameters were out of range, so the documentation available doesn't point to a specific cause — this needs manual review rather than a guessed hypothesis.
6. Structure your response with exactly these four sections, in this order, using these exact headings:
   ## What was detected
   ## Likely connection to machine context
   ## Recommended action and priority
   ## Sources
   The bracketed numbers (like [2]) shown before each excerpt under "Reference documentation" below are citation markers for THAT numbered list only. Cite documentation by its number wherever you draw on it inline (e.g. if an excerpt begins "[2] Source: ...", write [2]). In the "Sources" section, do NOT list bare numbers — for each one, write the number followed by that excerpt's title and section, copied from its "Source: ... — Section: ..." line, e.g.:
   ## Sources
   [1] Example Reference Document — Example Section
   [2] Example Report — Example Section
   (The two lines above are only a FORMAT example — never copy their titles into a real answer; use the actual "Source: ... — Section: ..." line from each numbered excerpt under "Reference documentation" below instead.)
   A reader seeing [2] inline must be able to find "[2] <title> — <section>" in the Sources section and know immediately what it refers to, without cross-referencing anything else. The two must match exactly: every [n] you cite inline appears in Sources with its title, and Sources contains no number you didn't actually cite inline — no extra entries, no missing ones. On an approved verdict (rule 1), that usually means Sources stays empty — cite documentation only if you're actually drawing on it for the preventive process note, never as a defect explanation. The "Machine context" section above is NOT part of that numbered list and has no number of its own: refer to it by plain description instead (e.g. "the elevated vibration reading" or "hours_since_service being 254.8h over the normal range"), never with a bracketed number, and never write the literal text "[n]" — there is no marker named "n" to fill in; that placeholder should never appear in your answer."""


def _format_detection(prediction: PredictionResult, defect_class: str | None) -> str:
    lines = [f"Verdict: {prediction.verdict}", f"Anomaly score: {prediction.score:.4f}"]
    if prediction.threshold is not None:
        lines.append(f"Threshold: {prediction.threshold:.4f}")
    lines.append(f"Detected defect class: {defect_class if defect_class else 'none'}")
    return "\n".join(lines)


def _format_machine_context(machine_context: MachineContext) -> str:
    if machine_context.anomalies:
        return "\n".join(f"- {a.describe()}" for a in machine_context.anomalies)
    if machine_context.reading is not None:
        # A reading exists and nothing was out of range — that's a real,
        # informative result (see rule 3 below), distinct from having no
        # data at all. Saying so plainly is what lets the prompt honestly
        # instruct the model to treat it as evidence pointing AWAY from
        # machine settings, not just an absence of information.
        return "All monitored machine parameters were within their normal range for this inspection."
    return "No machine parameter data was available for this inspection."


def _format_chunks(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No relevant documentation was retrieved for this component."
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] Source: {chunk.source} — Section: {chunk.section} (doc_type={chunk.doc_type})\n{chunk.text}"
        )
    return "\n\n".join(parts)


def build_prompt(
    component_name: str,
    prediction: PredictionResult,
    machine_context: MachineContext,
    chunks: list[RetrievedChunk],
    *,
    defect_class: str | None = None,
) -> str:
    """Assemble the full prompt. Called only when `chunks` is non-empty —
    reporter.py's "no relevant documentation" path never reaches here
    (see its own docstring for why that's a separate, unchanged branch).
    """
    return (
        f"{_INSTRUCTIONS}\n\n"
        f"=== Component ===\n{component_name}\n\n"
        f"=== Detection result ===\n{_format_detection(prediction, defect_class)}\n\n"
        f"=== Machine context ===\n{_format_machine_context(machine_context)}\n\n"
        f"=== Reference documentation ===\n{_format_chunks(chunks)}\n\n"
        "Write the maintenance note now, following the four-section structure from the rules above."
    )
