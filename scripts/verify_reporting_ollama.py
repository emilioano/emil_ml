"""Fas 4, step 5: same scenarios as verify_reporting_generation.py, but
with a real Ollama call (llm_mode="ollama") instead of mock — for judging
actual generation quality and grounding discipline, now that mock mode
has already verified the orchestration itself is correct.

Run with: python scripts/verify_reporting_ollama.py
(requires a local Ollama instance serving DEFAULT_RAG_LLM_MODEL, see
config/settings.py)
"""

from __future__ import annotations

import shutil
import sys

# Windows terminals default to a non-UTF-8 codepage that mangles non-ASCII
# characters (em dashes, curly quotes, ...) on print() otherwise — the
# underlying text is unaffected, this only fixes how this script displays it.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from emil_ml.config.logging_config import configure_logging
from emil_ml.config.registry import ComponentRegistry
from emil_ml.core.base import PredictionResult
from emil_ml.core.reporting import reporter
from emil_ml.core.reporting.machine_context.source import SqliteMachineContextSource
from emil_ml.utils.paths import for_component


def _print_report(report) -> None:  # noqa: ANN001 - reporter.ReportResult
    print(f"model: {report.model}")
    print(f"sources: {report.sources}")
    print(f"machine_context_used: {report.machine_context_used}")
    print()
    print("report_text:")
    print(report.report_text)


def main() -> None:
    configure_logging()
    registry = ComponentRegistry()
    source = SqliteMachineContextSource()
    tandborste = registry.get("tandborste")
    if tandborste is None:
        raise SystemExit("Component 'tandborste' not found — run earlier phases' setup first.")

    print("=" * 78)
    print("Scenario 1: classified defect (YOLO-style 'missing bristles') + injected temperature anomaly")
    print("=" * 78)
    source.insert_reading("tandborste", {"temperature": 88.0, "speed": 50.0, "pressure": 100.0,
                                          "vibration": 2.0, "hours_since_service": 100.0})
    prediction_classified = PredictionResult(
        verdict="failed",
        score=0.91,
        threshold=0.5,
        details={"detections": [{"class": "missing bristles", "confidence": 0.91, "box": [1, 2, 3, 4]}]},
    )
    report1 = reporter.generate_report(tandborste, prediction_classified, llm_mode="ollama")
    _print_report(report1)
    print()

    print("=" * 78)
    print("Scenario 2: unclassified (autoencoder-style, no defect class) + injected vibration anomaly")
    print("=" * 78)
    source.insert_reading("tandborste", {"temperature": 68.0, "speed": 50.0, "pressure": 100.0,
                                          "vibration": 7.5, "hours_since_service": 100.0})
    prediction_unclassified = PredictionResult(verdict="failed", score=0.045, threshold=0.032, details={})
    report2 = reporter.generate_report(tandborste, prediction_unclassified, llm_mode="ollama")
    _print_report(report2)
    print()

    print("=" * 78)
    print("Scenario 3: no relevant documentation available (confirms the honest path still applies with Ollama active)")
    print("=" * 78)
    empty_name = None
    try:
        from emil_ml.training import onboard

        empty_component = onboard.create_component("Fas4 No Docs Test Ollama", model_type="autoencoder", registry=registry)
        empty_name = empty_component.name
        prediction_empty = PredictionResult(verdict="failed", score=0.9, threshold=0.5, details={})
        report3 = reporter.generate_report(empty_component, prediction_empty, llm_mode="ollama")
        _print_report(report3)
        ok3 = report3.prompt_used is None and report3.model is None and "No relevant documentation" in report3.report_text
        print()
        print(f"-> {'PASS' if ok3 else 'FAIL'}: honest 'no documentation' path not bypassed by Ollama being active.")
    finally:
        if empty_name:
            registry.delete(empty_name)
            shutil.rmtree(for_component(empty_name).root, ignore_errors=True)


if __name__ == "__main__":
    main()
