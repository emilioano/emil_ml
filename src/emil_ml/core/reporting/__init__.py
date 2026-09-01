"""RAG-based maintenance report generation — a step that runs AFTER detection.

Not a model_type: it never appears in VALID_MODEL_TYPES or
registry_factory's dispatch tables. It takes a `PredictionResult` (from
whichever method produced it — autoencoder, classifier, YOLO, isolation
forest, or PatchCore all produce the same shape, see core/base.py) and
produces a `ReportResult`. Nothing in here branches on model_type; a report
is built the same way regardless of which detector ran.

Three separated responsibilities, deliberately not mixed:
- retrieve (knowledge/retriever.py): metadata-filtered similarity search
  over a document knowledge base, returns chunks with provenance.
- formulate (prompt.py): turns a detection result + retrieved chunks +
  machine-context anomalies into a prompt. No I/O, no model calls.
- generate (llm.py): sends a prompt to a swappable LLM backend (mock /
  Ollama / cloud) and returns text. Doesn't know or care how the prompt
  was built.

reporter.py is the only place that wires these together, plus a fourth
input this project treats as equally important: machine_context/ —
production-line parameters (temperature, vibration, ...) at inspection
time. This matters most exactly when the detector's own output is weakest:
PatchCore and the autoencoder report "something's off here" with no defect
label, so machine-context anomalies (e.g. "12° over normal" ->
"over-temperature") are often the only concrete lead a generated report
has to point to. A report is always phrased as a hypothesis ("likely
cause / worth investigating"), never asserted as settled causality — the
detector found an anomaly; it did not diagnose a root cause.

Submodules:
    knowledge/         Document indexing (indexer.py) and retrieval
                        (retriever.py) — the vector-store side.
    machine_context/    Fetches and interprets production-line parameters
                        for a given inspection (source.py, analyzer.py).
    prompt.py           Assembles model inputs into a single prompt string.
    llm.py              Swappable text-generation backend.
    reporter.py         Orchestrates the above into a ReportResult.
"""
