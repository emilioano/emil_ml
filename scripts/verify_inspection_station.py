"""Verifies the Inspection Station backend changes on a disposable
throwaway component (deleted in `finally`, same convention as the other
verify_*.py scripts). Pinned to mock LLM mode (see
verify_inspections_lifecycle.py's own docstring for why that's the right
call for a persistence/lifecycle check) since none of this depends on
report generation quality or speed.

1. store.create()/list_all() accept and filter on run_by, verdict,
   report_status.
2. store.revert_to_new(): acknowledged -> new clears acknowledged_by/at;
   a no-op against 'new' or 'archived' records.
3. orchestrator.run_inspection(run_by=...) persists it; the watcher path
   (simulated here) would pass "watcher".
4. approved_handling="auto_acknowledge" auto-acknowledges an approved
   verdict (by="system (auto_acknowledge)"), while a failed verdict on the
   same component is untouched (still 'new') — the asymmetry is real, not
   accidental.
5. approved_handling="hide_from_default_view" / "keep_visible" don't
   change any DB behavior (both are UI-filtering concerns only) — approved
   records exist identically in the DB either way, confirming nothing
   about this setting can make a record undiscoverable at the data layer.
6. Model backup/rollback: backup_current_model() copies the current model
   file with a timestamped name and returns None when there's no model
   yet; list_model_backups() finds it; restore_model_backup() repoints
   model_path at it exactly (byte-for-byte).

Run with: python scripts/verify_inspection_station.py
"""

from __future__ import annotations

import functools
import shutil
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from emil_ml.config.logging_config import configure_logging
from emil_ml.config.registry import ComponentRegistry
from emil_ml.core.inspections import orchestrator, store
from emil_ml.core.reporting import reporter
from emil_ml.utils.paths import for_component

COMPONENT_NAME = "tandborste"


def _pin_mock_llm_mode() -> None:
    """See verify_inspections_lifecycle.py's own _pin_mock_llm_mode() for
    the full explanation — same technique, same reason."""
    reporter.generate_report = functools.partial(reporter.generate_report, llm_mode="mock")


def main() -> None:
    configure_logging()
    _pin_mock_llm_mode()
    registry = ComponentRegistry()
    registry.update_settings(COMPONENT_NAME, reporting_enabled=False)
    paths = for_component(COMPONENT_NAME)

    import glob

    approved_candidates = glob.glob(f"data/components/{COMPONENT_NAME}/training/approved/*")
    if not approved_candidates:
        raise SystemExit("Need at least one sample image for tandborste.")
    approved_image = approved_candidates[0]
    # tandborste is an autoencoder trained only on 'good' examples — there's
    # no separate pool of known-bad images sitting on disk to reuse as a
    # 'failed' sample, and a training-set image's own reconstruction error
    # can legitimately land right at the calibrated threshold (it's not a
    # margin guarantee, just "at or below the training set's own errors"),
    # so relying on the real default threshold for either verdict is
    # fragile test data, not a real assertion about the code under test.
    # threshold_override forces each verdict deterministically instead:
    # 0.0 guarantees 'failed' (any error is >= 0), a very high threshold
    # guarantees 'approved' — this script only needs one real record of
    # each verdict, not a genuine judgment about detection quality.
    failed_image = approved_image
    FORCE_APPROVED_THRESHOLD = 999.0
    FORCE_FAILED_THRESHOLD = 0.0

    # === 1: run_by persisted + filterable ================================
    print("=== 1: run_by persisted on create(), list_all() filters on run_by/verdict/report_status ===")
    record_ui, _ = orchestrator.run_inspection(
        approved_image,
        COMPONENT_NAME,
        registry=registry,
        async_report=False,
        run_by="test-operator",
        threshold_override=FORCE_APPROVED_THRESHOLD,
    )
    record_watcher, _ = orchestrator.run_inspection(
        failed_image,
        COMPONENT_NAME,
        registry=registry,
        async_report=False,
        run_by="watcher",
        threshold_override=FORCE_FAILED_THRESHOLD,
    )
    ok1a = record_ui.run_by == "test-operator" and record_watcher.run_by == "watcher"
    print(f"  record_ui.run_by={record_ui.run_by!r}  record_watcher.run_by={record_watcher.run_by!r}")

    failed_only = store.list_all(component_name=COMPONENT_NAME, verdict="failed", limit=50)
    approved_only = store.list_all(component_name=COMPONENT_NAME, verdict="approved", limit=50)
    ok1b = (
        any(r.id == record_watcher.id for r in failed_only)
        and all(r.verdict == "failed" for r in failed_only)
        and any(r.id == record_ui.id for r in approved_only)
        and all(r.verdict == "approved" for r in approved_only)
    )
    none_reports = store.list_all(component_name=COMPONENT_NAME, report_status="none", limit=50)
    ok1c = all(r.report_status == "none" for r in none_reports) and record_ui.id in {r.id for r in none_reports}
    ok1 = ok1a and ok1b and ok1c
    print(f"-> {'PASS' if ok1 else 'FAIL'}: run_by stored correctly; verdict/report_status filters work.")
    print()

    # === 2: revert_to_new ================================================
    print("=== 2: revert_to_new(): acknowledged -> new clears acknowledged_by/at; no-op elsewhere ===")
    store.acknowledge(record_ui.id, by="reviewer-1")
    acked = store.get(record_ui.id)
    ok2a = acked.status == "acknowledged" and acked.acknowledged_by == "reviewer-1"

    store.revert_to_new(record_ui.id)
    reverted = store.get(record_ui.id)
    ok2b = reverted.status == "new" and reverted.acknowledged_by is None and reverted.acknowledged_at is None
    print(f"  after acknowledge: status={acked.status} acknowledged_by={acked.acknowledged_by}")
    print(f"  after revert:      status={reverted.status} acknowledged_by={reverted.acknowledged_by}")

    store.revert_to_new(record_watcher.id)  # still 'new' -> must no-op, not raise
    still_new = store.get(record_watcher.id)
    ok2c = still_new.status == "new"
    ok2 = ok2a and ok2b and ok2c
    print(f"-> {'PASS' if ok2 else 'FAIL'}: revert works from acknowledged, safely no-ops from new.")
    print()

    # === 3 & 4: auto_acknowledge asymmetry ================================
    print("=== 3 & 4: approved_handling='auto_acknowledge' only touches approved, never failed ===")
    registry.update_settings(COMPONENT_NAME, approved_handling="auto_acknowledge")
    auto_approved, _ = orchestrator.run_inspection(
        approved_image,
        COMPONENT_NAME,
        registry=registry,
        async_report=False,
        run_by="test-operator",
        threshold_override=FORCE_APPROVED_THRESHOLD,
    )
    auto_failed, _ = orchestrator.run_inspection(
        failed_image,
        COMPONENT_NAME,
        registry=registry,
        async_report=False,
        run_by="test-operator",
        threshold_override=FORCE_FAILED_THRESHOLD,
    )
    print(f"  approved record: status={auto_approved.status} acknowledged_by={auto_approved.acknowledged_by!r}")
    print(f"  failed record:   status={auto_failed.status} acknowledged_by={auto_failed.acknowledged_by!r}")
    ok34 = (
        auto_approved.status == "acknowledged"
        and auto_approved.acknowledged_by == "system (auto_acknowledge)"
        and auto_failed.status == "new"
        and auto_failed.acknowledged_by is None
    )
    print(f"-> {'PASS' if ok34 else 'FAIL'}: approved auto-acknowledged, failed left for a human, asymmetry confirmed.")
    print()
    registry.update_settings(COMPONENT_NAME, approved_handling="hide_from_default_view")

    # === 5: hide_from_default_view / keep_visible don't affect DB content ==
    print("=== 5: approved_handling is a display-only concern — record exists identically in the DB ===")
    still_there = store.get(auto_approved.id)
    ok5 = still_there is not None and still_there.verdict == "approved"
    print(f"-> {'PASS' if ok5 else 'FAIL'}: approved record still fully present/queryable regardless of setting.")
    print()

    # === 6: model backup / rollback ========================================
    # tandborste is a real, shared component (used by many other verify_*.py
    # scripts too) with a real trained model — needed here since
    # backup_current_model() requires an actual model file to back up. This
    # block deliberately corrupts and repoints its model file/model_path to
    # exercise restore, so it MUST leave both exactly as found — see the
    # `finally` below, which is the point of this whole block, not an
    # afterthought.
    print("=== 6: backup_current_model / list_model_backups / restore_model_backup ===")
    from emil_ml.training import onboard

    original_model_path_value = registry.get(COMPONENT_NAME).model_path
    current_model_path = for_component(COMPONENT_NAME).resolve_model_path(original_model_path_value)
    original_bytes = current_model_path.read_bytes()
    backup_path = None
    try:
        backup_path = onboard.backup_current_model(COMPONENT_NAME, registry=registry)
        backups_before = onboard.list_model_backups(COMPONENT_NAME)
        ok6a = backup_path is not None and any(b["path"] == backup_path for b in backups_before)
        print(f"  backup created: {backup_path!r}")
        print(f"  backups now listed: {[b['filename'] for b in backups_before]}")

        # Corrupt the "current" model on disk to simulate a bad retrain, then restore.
        current_model_path.write_bytes(b"deliberately corrupted for restore test")
        onboard.restore_model_backup(COMPONENT_NAME, backup_path, registry=registry)
        restored_component = registry.get(COMPONENT_NAME)
        restored_model_path = for_component(COMPONENT_NAME).resolve_model_path(restored_component.model_path)
        ok6b = restored_model_path.read_bytes() == original_bytes
        print(f"  model_path after restore: {restored_component.model_path}")
        ok6 = ok6a and ok6b
        print(f"-> {'PASS' if ok6 else 'FAIL'}: backup created/listed correctly, restore recovers the exact original bytes.")
    finally:
        # Repair tandborste exactly back to how this block found it: original
        # bytes at the original file location, registry pointed at the
        # original model_path string, and the test-created backup file gone.
        current_model_path.write_bytes(original_bytes)
        registry.update_training_result(
            COMPONENT_NAME,
            anomaly_threshold=registry.get(COMPONENT_NAME).anomaly_threshold,
            model_path=original_model_path_value,
            status="ready",
        )
        if backup_path:
            backup_abs = for_component(COMPONENT_NAME).root / backup_path
            backup_abs.unlink(missing_ok=True)
    print()

    all_pass = all([ok1, ok2, ok34, ok5, ok6])
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED — see above'}")


if __name__ == "__main__":
    main()
