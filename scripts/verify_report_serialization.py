"""Verifies report generation is serialized against Ollama contention —
core/inspections/report_worker.py's single-worker-queue replacing the old
one-thread-per-report design.

Pinned to mock LLM mode (see verify_inspections_lifecycle.py's own
_pin_mock_llm_mode() for the pattern), with an injected artificial delay
per report so overlapping execution (if it happened) would be observable —
this script tests the CONCURRENCY property, not real Ollama timing.

1. Fire several run_inspection() calls back-to-back. Each must return
   almost immediately (detection stays fast/unblocked) — nowhere near the
   per-report delay, proving detection never waits on the report queue.
2. While those reports are in flight, the max number running AT THE SAME
   TIME must be exactly 1 — proving reports are serialized against each
   other, not running concurrently.
3. Total wall time for every report to finish must be close to
   N * per_report_delay (serialized), not close to per_report_delay
   (which would mean they ran in parallel).

Run with: python scripts/verify_report_serialization.py
"""

from __future__ import annotations

import functools
import sys
import threading
import time

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from emil_ml.config.logging_config import configure_logging
from emil_ml.config.registry import ComponentRegistry
from emil_ml.core.inspections import orchestrator, store
from emil_ml.core.reporting import reporter

COMPONENT_NAME = "tandborste"
PER_REPORT_DELAY_SECONDS = 1.0
N_INSPECTIONS = 4

_lock = threading.Lock()
_in_flight = 0
_max_concurrent = 0
_call_count = 0


def _instrumented_generate_report(*args, **kwargs):
    global _in_flight, _max_concurrent, _call_count
    with _lock:
        _in_flight += 1
        _max_concurrent = max(_max_concurrent, _in_flight)
        _call_count += 1
    try:
        time.sleep(PER_REPORT_DELAY_SECONDS)  # simulate Ollama's real generation latency
        kwargs["llm_mode"] = "mock"
        return _real_generate_report(*args, **kwargs)
    finally:
        with _lock:
            _in_flight -= 1


def main() -> None:
    global _real_generate_report
    configure_logging()
    registry = ComponentRegistry()
    registry.update_settings(COMPONENT_NAME, reporting_enabled=True, reporting_condition="always")

    _real_generate_report = reporter.generate_report
    reporter.generate_report = _instrumented_generate_report

    try:
        import glob

        candidates = glob.glob(f"data/components/{COMPONENT_NAME}/training/approved/*")
        if not candidates:
            raise SystemExit("Need at least one sample image for tandborste.")
        sample_image = candidates[0]

        print(f"=== Firing {N_INSPECTIONS} run_inspection() calls back-to-back ===")
        records = []
        call_durations = []
        overall_start = time.time()
        for i in range(N_INSPECTIONS):
            t0 = time.time()
            record, _ = orchestrator.run_inspection(
                sample_image, COMPONENT_NAME, registry=registry, run_by="serialization-test"
            )
            call_durations.append(time.time() - t0)
            records.append(record)
        dispatch_time = time.time() - overall_start
        print(f"  all {N_INSPECTIONS} run_inspection() calls returned in {dispatch_time:.3f}s total")
        print(f"  individual call durations: {[f'{d:.3f}s' for d in call_durations]}")

        # The first call's duration is dominated by one-time predictor
        # cold-load (the autoencoder's Keras model lazily loads and caches
        # on first inference — nothing to do with report generation, and
        # the exact "cold start" overhead this project has hit repeatedly
        # elsewhere). What actually proves detection never blocks on the
        # report queue is every call AFTER warmup staying fast — nowhere
        # near PER_REPORT_DELAY_SECONDS, even with 1-3 reports already
        # queued ahead of them.
        warm_durations = call_durations[1:]
        ok_fast_dispatch = all(d < PER_REPORT_DELAY_SECONDS for d in warm_durations)
        print(f"  post-warmup call durations: {[f'{d:.3f}s' for d in warm_durations]} (each must stay under {PER_REPORT_DELAY_SECONDS}s)")
        print(f"-> {'PASS' if ok_fast_dispatch else 'FAIL'}: detection/dispatch stayed fast, never blocked on the report queue.")
        print()

        print("=== Waiting for all reports to complete ===")
        # Generous margin: each report also does REAL retrieval (a real
        # Ollama query-embedding call against tandborste's actual indexed
        # knowledge base) on top of the artificial per-report sleep above —
        # this budget just needs to comfortably outlast N serialized
        # reports, it isn't itself part of what's being measured.
        deadline = time.time() + (PER_REPORT_DELAY_SECONDS + 5) * N_INSPECTIONS + 30
        while time.time() < deadline:
            statuses = [store.get(r.id).report_status for r in records]
            if all(s in ("complete", "failed") for s in statuses):
                break
            time.sleep(0.1)
        total_wall_time = time.time() - overall_start
        final_statuses = [store.get(r.id).report_status for r in records]
        print(f"  final report statuses: {final_statuses}")
        print(f"  total wall time until all reports finished: {total_wall_time:.3f}s")
        print()

        print("=== Checking concurrency ===")
        print(f"  max concurrent report generations observed: {_max_concurrent} (expected 1)")
        print(f"  total report calls made: {_call_count} (expected {N_INSPECTIONS})")
        ok_serialized = _max_concurrent == 1
        print(f"-> {'PASS' if ok_serialized else 'FAIL'}: reports never ran concurrently — serialized against each other.")
        print()

        expected_serial_time = PER_REPORT_DELAY_SECONDS * N_INSPECTIONS
        ok_timing = total_wall_time >= expected_serial_time * 0.9
        print(
            f"  total time ({total_wall_time:.3f}s) vs. expected serialized time "
            f"(~{expected_serial_time:.1f}s for {N_INSPECTIONS} x {PER_REPORT_DELAY_SECONDS}s)"
        )
        print(f"-> {'PASS' if ok_timing else 'FAIL'}: timing matches serialized (not parallel) execution.")
        print()

        all_pass = ok_fast_dispatch and all(s == "complete" for s in final_statuses) and ok_serialized and ok_timing
        print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED — see above'}")
    finally:
        reporter.generate_report = _real_generate_report
        # Cleanup: remove the test inspection rows + their image files.
        from emil_ml.config.database import connect
        from emil_ml.config.settings import DB_PATH, INSPECTIONS_TABLE
        from emil_ml.utils.paths import for_component

        paths = for_component(COMPONENT_NAME)
        ids = [r.id for r in records] if "records" in dir() else []
        for i in ids:
            record = store.get(i)
            if record is None:
                continue
            for rel_path in (record.image_path, record.report_path):
                if rel_path:
                    (paths.root / rel_path).unlink(missing_ok=True)
        if ids:
            with connect(DB_PATH) as conn:
                placeholders = ", ".join("?" * len(ids))
                conn.execute(f"DELETE FROM {INSPECTIONS_TABLE} WHERE id IN ({placeholders})", ids)


if __name__ == "__main__":
    main()
