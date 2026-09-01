"""Verifies bulk acknowledge/archive on a disposable throwaway component
(deleted in `finally`, same convention as the other verify_*.py scripts).

1. store.bulk_acknowledge(): only flips currently-'new' rows, in one
   batch; already-acknowledged/archived ids are silently skipped: the
   returned count reflects only what actually changed.
2. lifecycle.bulk_archive_approved(): only 'approved' verdicts are ever
   archived — a 'failed' record in the selection is skipped and reported,
   never archived; already-archived records are skipped (idempotent /
   resumable — calling it twice with the same ids never double-processes
   or errors); a still-'new' approved record is acknowledged first
   automatically.
3. An archived approved record is still fully present and queryable in
   the DB afterward — archiving is never deletion.
4. store.unverify(): resets a verified record back to 'unverified',
   clearing the label/by/at/error_type and resetting
   verified_incorporation_status back to 'pending'.

Run with: python scripts/verify_bulk_actions.py
"""

from __future__ import annotations

import glob
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from emil_ml.config.logging_config import configure_logging
from emil_ml.config.registry import ComponentRegistry
from emil_ml.core.inspections import lifecycle, orchestrator, store
from emil_ml.utils.paths import for_component

COMPONENT_NAME = "tandborste"
FORCE_APPROVED_THRESHOLD = 999.0
FORCE_FAILED_THRESHOLD = -1.0


def main() -> None:
    configure_logging()
    registry = ComponentRegistry()
    registry.update_settings(COMPONENT_NAME, reporting_enabled=False, approved_handling="hide_from_default_view")
    paths = for_component(COMPONENT_NAME)

    candidates = glob.glob(f"data/components/{COMPONENT_NAME}/training/approved/*")
    if not candidates:
        raise SystemExit("Need at least one sample image for tandborste.")
    sample_image = candidates[0]

    created_ids: list[int] = []

    def _inspect(threshold: float) -> int:
        record, _ = orchestrator.run_inspection(
            sample_image, COMPONENT_NAME, registry=registry, async_report=False,
            run_by="bulk-test", threshold_override=threshold,
        )
        created_ids.append(record.id)
        return record.id

    try:
        # === Setup: a mix of approved (new), approved (already acknowledged),
        # failed (new), and one approved that will be pre-archived. =========
        approved_new_ids = [_inspect(FORCE_APPROVED_THRESHOLD) for _ in range(4)]
        approved_already_acked_id = _inspect(FORCE_APPROVED_THRESHOLD)
        store.acknowledge(approved_already_acked_id, by="pre-existing")
        failed_id = _inspect(FORCE_FAILED_THRESHOLD)

        # === 1: bulk_acknowledge — only 'new' rows flip, count is exact ===
        print("=== 1: bulk_acknowledge() only flips 'new' rows ===")
        target_ids = approved_new_ids + [approved_already_acked_id, failed_id]
        count = store.bulk_acknowledge(target_ids, by="bulk-operator")
        reloaded = {i: store.get(i) for i in target_ids}
        ok1 = (
            count == 5  # 4 approved_new + failed_id (both were 'new'); already_acked was NOT 'new'
            and all(reloaded[i].status == "acknowledged" and reloaded[i].acknowledged_by == "bulk-operator" for i in approved_new_ids)
            and reloaded[failed_id].status == "acknowledged"
            and reloaded[approved_already_acked_id].acknowledged_by == "pre-existing"  # untouched, was already acked
        )
        print(f"  bulk_acknowledge returned count={count} (expected 5)")
        print(f"-> {'PASS' if ok1 else 'FAIL'}")
        print()

        # === 2: bulk_archive_approved — approved-only hard guard =============
        print("=== 2: bulk_archive_approved() archives only approved, skips failed, is resumable ===")
        archive_selection = approved_new_ids + [failed_id]  # deliberately includes the failed one
        progress_calls = []
        result = lifecycle.bulk_archive_approved(
            archive_selection, by="bulk-operator", on_progress=lambda d, t: progress_calls.append((d, t))
        )
        print(f"  result: archived={result.archived} skipped_failed_verdict={result.skipped_failed_verdict} "
              f"skipped_already_archived={result.skipped_already_archived} errors={result.errors}")
        ok2a = (
            result.archived == 4
            and result.skipped_failed_verdict == 1
            and result.skipped_already_archived == 0
            and not result.errors
            and progress_calls[-1] == (5, 5)  # on_progress fires once per input id, including skipped ones
        )
        reloaded_failed = store.get(failed_id)
        ok2b = reloaded_failed.status == "acknowledged" and reloaded_failed.image_path is not None  # untouched by archive
        print(f"  failed record after bulk archive attempt: status={reloaded_failed.status} (must NOT be 'archived')")
        print(f"-> {'PASS' if ok2a and ok2b else 'FAIL'}: only approved archived, failed left alone.")
        print()

        # === Resumability: re-run the exact same selection =====================
        print("=== Resumability: re-running bulk_archive_approved() with the SAME ids is a safe no-op ===")
        second_result = lifecycle.bulk_archive_approved(archive_selection, by="bulk-operator")
        ok_resume = (
            second_result.archived == 0
            and second_result.skipped_already_archived == 4
            and second_result.skipped_failed_verdict == 1
        )
        print(f"  second call result: {second_result}")
        print(f"-> {'PASS' if ok_resume else 'FAIL'}: already-archived records are skipped, never double-processed.")
        print()

        # === 3: archived approved records are still fully queryable ============
        print("=== 3: archiving is never deletion — archived approved records stay in the DB ===")
        still_there = [store.get(i) for i in approved_new_ids]
        ok3 = all(r is not None and r.status == "archived" and r.verdict == "approved" for r in still_there)
        for r in still_there:
            print(f"  id={r.id} status={r.status} image_path={r.image_path}")
        print(f"-> {'PASS' if ok3 else 'FAIL'}: every archived approved record is still fully present and readable.")
        print()

        # === 4: unverify() undoes a mistaken verify() click =====================
        print("=== 4: unverify() resets verified_* back to defaults ===")
        mistake_id = approved_already_acked_id
        label = {"verdict": "approved", "defect_classes": [], "boxes": []}
        store.verify(mistake_id, status="verified_correct", label=label, by="qa-reviewer")
        before_undo = store.get(mistake_id)
        store.unverify(mistake_id)
        after_undo = store.get(mistake_id)
        ok4 = (
            before_undo.verified_status == "verified_correct"
            and after_undo.verified_status == "unverified"
            and after_undo.verified_label is None
            and after_undo.verified_by is None
            and after_undo.verified_at is None
            and after_undo.verified_error_type is None
            and after_undo.verified_incorporation_status == "pending"
            and after_undo.status == before_undo.status  # workflow status untouched by unverify()
        )
        print(f"  before undo: verified_status={before_undo.verified_status}")
        print(f"  after undo:  verified_status={after_undo.verified_status} label={after_undo.verified_label}")
        print(f"-> {'PASS' if ok4 else 'FAIL'}")
        print()

        all_pass = all([ok1, ok2a, ok2b, ok_resume, ok3, ok4])
        print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED — see above'}")
    finally:
        registry.update_settings(COMPONENT_NAME, approved_handling="hide_from_default_view")
        # Test-only cleanup: tandborste is a real, shared component other
        # verify_*.py scripts also use, with real historical analyzed/
        # archive/ files of its own — remove ONLY the specific files this
        # run's own inspection ids point to, never the whole directory
        # (that would destroy other scripts' legitimate data).
        from emil_ml.config.database import connect
        from emil_ml.config.settings import DB_PATH, INSPECTIONS_TABLE

        for i in created_ids:
            record = store.get(i)
            if record is None:
                continue
            for rel_path in (record.image_path, record.report_path):
                if rel_path:
                    (paths.root / rel_path).unlink(missing_ok=True)
        if created_ids:
            with connect(DB_PATH) as conn:
                placeholders = ", ".join("?" * len(created_ids))
                conn.execute(f"DELETE FROM {INSPECTIONS_TABLE} WHERE id IN ({placeholders})", created_ids)


if __name__ == "__main__":
    main()
