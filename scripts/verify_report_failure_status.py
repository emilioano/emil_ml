"""Verifies a degraded report generation (Ollama unreachable, empty
response, ...) persists as report_status='failed', not 'complete' — the
bug this fixes: llm.generate() degrades honestly (returns an LLMResult
with `error` set, never raises), but reporter.ReportResult silently
dropped that flag and orchestrator.py always wrote report_status=
'complete' whenever no Python exception occurred, so a report whose
entire body was an apology that Ollama couldn't be reached showed up on
the Inspection Station as "Report available" — indistinguishable from a
real one until you opened it.

Run with: python scripts/verify_report_failure_status.py
"""

from __future__ import annotations

import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from emil_ml.config.logging_config import configure_logging
from emil_ml.config.registry import ComponentRegistry
from emil_ml.core.inspections import orchestrator, store
from emil_ml.core.reporting import llm

COMPONENT_NAME = "tandborste"


def main() -> None:
    configure_logging()
    registry = ComponentRegistry()
    registry.update_settings(COMPONENT_NAME, reporting_enabled=True, reporting_condition="always")

    real_generate = llm.generate

    def _degraded_generate(*args, **kwargs):
        return llm.LLMResult(
            text="Report generation failed: could not reach Ollama at http://localhost:11434 (simulated).",
            model=kwargs.get("model") or "qwen3:8b",
            error="simulated connection failure",
        )

    llm.generate = _degraded_generate
    record = None
    try:
        import glob

        candidates = glob.glob(f"data/components/{COMPONENT_NAME}/training/approved/*")
        if not candidates:
            raise SystemExit("Need at least one sample image for tandborste.")

        print("=== Running an inspection with a simulated degraded Ollama response ===")
        record, _ = orchestrator.run_inspection(
            candidates[0], COMPONENT_NAME, registry=registry, async_report=False, run_by="failure-status-test"
        )
        print(f"  report_status: {record.report_status}")
        print(f"  report_text: {record.report_text!r}")

        ok = (
            record.report_status == "failed"
            and record.report_text is not None
            and "could not reach Ollama" in record.report_text
        )
        print(f"-> {'PASS' if ok else 'FAIL'}: degraded generation persists as report_status='failed', with the honest error text kept.")
    finally:
        llm.generate = real_generate
        if record is not None:
            from emil_ml.config.database import connect
            from emil_ml.config.settings import DB_PATH, INSPECTIONS_TABLE
            from emil_ml.utils.paths import for_component

            paths = for_component(COMPONENT_NAME)
            reloaded = store.get(record.id)
            if reloaded is not None:
                for rel_path in (reloaded.image_path, reloaded.report_path):
                    if rel_path:
                        (paths.root / rel_path).unlink(missing_ok=True)
            with connect(DB_PATH) as conn:
                conn.execute(f"DELETE FROM {INSPECTIONS_TABLE} WHERE id = ?", (record.id,))


if __name__ == "__main__":
    main()
