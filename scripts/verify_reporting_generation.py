"""Fas 4 sanity check: full orchestration (machine context -> retrieval ->
prompt -> generation -> ReportResult) in mock LLM mode first, so
orchestration correctness can be verified independently of generation
quality — same "isolate before combining" discipline as every earlier
phase.

Scenarios:
1. Classified defect (simulated YOLO-style detection) + injected machine
   anomaly, on tandborste.
2. Unclassified case (simulated autoencoder-style prediction, no defect
   class) + injected machine anomaly, on tandborste — confirms the
   prompt leans on machine context when there's no defect label.
3. No relevant documentation available — confirms the pre-Fas-4 "nothing
   found" branch is untouched and the LLM is never called.

Run with: python scripts/verify_reporting_generation.py
"""

from __future__ import annotations

import shutil
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from emil_ml.config.logging_config import configure_logging
from emil_ml.config.registry import ComponentRegistry
from emil_ml.core.base import PredictionResult
from emil_ml.core.reporting import reporter
from emil_ml.core.reporting.machine_context.source import SqliteMachineContextSource
from emil_ml.utils.paths import for_component


def _print_report(report) -> None:  # noqa: ANN001 - reporter.ReportResult
    print("report_text:")
    print(report.report_text)
    print()
    print("sources:", report.sources)
    print("machine_context_used:", report.machine_context_used)
    print("model:", report.model)
    print()
    if report.prompt_used:
        print("prompt_used:")
        print("-" * 70)
        print(report.prompt_used)
        print("-" * 70)
    else:
        print("prompt_used: None")


def main() -> None:
    configure_logging()
    registry = ComponentRegistry()
    source = SqliteMachineContextSource()
    tandborste = registry.get("tandborste")
    if tandborste is None:
        raise SystemExit("Component 'tandborste' not found — run earlier phases' setup first.")

    # === Scenario 1: classified defect + injected machine anomaly ==========
    print("=" * 78)
    print("Scenario 1: classified defect (YOLO-style) + injected temperature anomaly")
    print("=" * 78)
    source.insert_reading("tandborste", {"temperature": 88.0, "speed": 50.0, "pressure": 100.0,
                                          "vibration": 2.0, "hours_since_service": 100.0})
    prediction_classified = PredictionResult(
        verdict="failed",
        score=0.91,
        threshold=0.5,
        details={"detections": [{"class": "missing bristles", "confidence": 0.91, "box": [1, 2, 3, 4]}]},
    )
    report1 = reporter.generate_report(tandborste, prediction_classified, llm_mode="mock")
    _print_report(report1)
    ok1 = (
        report1.machine_context_used == ["over-temperature"]
        and len(report1.sources) > 0
        and report1.prompt_used is not None
        and "missing bristles" in report1.prompt_used
        and "over-temperature" in report1.prompt_used.lower()
        and "Detected defect class: missing bristles" in report1.prompt_used
    )
    print(f"-> {'PASS' if ok1 else 'FAIL'}: prompt contains defect class + machine anomaly, sources present.")
    print()

    # === Scenario 2: unclassified case + injected machine anomaly ==========
    print("=" * 78)
    print("Scenario 2: unclassified (autoencoder-style, no defect class) + injected vibration anomaly")
    print("=" * 78)
    source.insert_reading("tandborste", {"temperature": 68.0, "speed": 50.0, "pressure": 100.0,
                                          "vibration": 7.5, "hours_since_service": 100.0})
    prediction_unclassified = PredictionResult(verdict="failed", score=0.045, threshold=0.032, details={})
    report2 = reporter.generate_report(tandborste, prediction_unclassified, llm_mode="mock")
    _print_report(report2)
    ok2 = (
        report2.machine_context_used == ["elevated vibration"]
        and report2.prompt_used is not None
        and "Detected defect class: none" in report2.prompt_used
        and "elevated vibration" in report2.prompt_used.lower()
    )
    print(f"-> {'PASS' if ok2 else 'FAIL'}: prompt shows no defect class and leans on machine context.")
    print()

    # === Scenario 3: no relevant documentation available ===================
    print("=" * 78)
    print("Scenario 3: component with reporting-relevant setup but no indexed documentation")
    print("=" * 78)
    empty_name = None
    try:
        from emil_ml.training import onboard

        empty_component = onboard.create_component("Fas4 No Docs Test", model_type="autoencoder", registry=registry)
        empty_name = empty_component.name
        prediction_empty = PredictionResult(verdict="failed", score=0.9, threshold=0.5, details={})
        report3 = reporter.generate_report(empty_component, prediction_empty, llm_mode="mock")
        _print_report(report3)
        ok3 = (
            report3.sources == []
            and report3.prompt_used is None
            and report3.model is None
            and "No relevant documentation" in report3.report_text
        )
        print(f"-> {'PASS' if ok3 else 'FAIL'}: honest 'no documentation' report, LLM never called (prompt_used/model are None).")
    finally:
        if empty_name:
            registry.delete(empty_name)
            shutil.rmtree(for_component(empty_name).root, ignore_errors=True)

    print()
    print("Fas 4 mock-mode orchestration check complete.")


if __name__ == "__main__":
    main()
