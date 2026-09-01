"""Verifies the intersection between the correction feedback loop and the
archive/retention lifecycle, on the shared 'tandborste' component (its
own inspection rows cleaned up afterward — see the other verify_*.py
scripts that touch tandborste for the same convention).

1. list_verified_for_training() has no hidden lifecycle filter: a
   verified inspection that gets archived is still returned.
2. cleanup_archived_inspections() never deletes an archived inspection
   with a verified label whose verified_incorporation_status is still
   'pending', no matter how old it is — but DOES delete it once marked
   'incorporated' (protection lifted, value already realized).
3. A plain unverified archived inspection is deleted by retention exactly
   as before (the protection is additive, not a general behavior change).
4. A record that's old enough by verdict but not yet past the retention
   cutoff is left alone regardless of verification state.
5. retention.pending_verified_counts() surfaces the protected record —
   the visibility half of "logged/shown, not silently accumulating".

Run with: python scripts/verify_retention_protection.py
"""

from __future__ import annotations

import glob
import sys
from datetime import datetime, timedelta, timezone

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from emil_ml.config.database import connect
from emil_ml.config.logging_config import configure_logging
from emil_ml.config.registry import ComponentRegistry
from emil_ml.config.settings import DB_PATH, INSPECTIONS_TABLE
from emil_ml.core.inspections import lifecycle, orchestrator, retention, store
from emil_ml.utils.paths import for_component

COMPONENT_NAME = "tandborste"
FORCE_APPROVED_THRESHOLD = 999.0


def _force_archived_at(inspection_id: int, when: datetime) -> None:
    """Test-only: mark_archived() always stamps archived_at = now(), so
    the only way to test age-based retention logic deterministically is
    to backdate it directly after the fact."""
    with connect(DB_PATH) as conn:
        conn.execute(
            f"UPDATE {INSPECTIONS_TABLE} SET archived_at = ? WHERE id = ?",
            (when.strftime("%Y-%m-%d %H:%M:%S"), inspection_id),
        )


def _archive(record_id: int, paths) -> None:
    store.acknowledge(record_id, by="tester")
    record = store.get(record_id)
    new_image_path, new_report_path = lifecycle.archive(paths, record)
    store.mark_archived(record_id, image_path=new_image_path, report_path=new_report_path)


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
    old_retention_days = registry.get(COMPONENT_NAME).inspection_retention_days

    def _inspect() -> int:
        record, _ = orchestrator.run_inspection(
            sample_image, COMPONENT_NAME, registry=registry, async_report=False,
            run_by="retention-test", threshold_override=FORCE_APPROVED_THRESHOLD,
        )
        created_ids.append(record.id)
        return record.id

    try:
        registry.update_settings(COMPONENT_NAME, inspection_retention_days=1)

        now = datetime.now(timezone.utc)
        ten_days_ago = now - timedelta(days=10)

        # A: unverified, archived long ago -> should be deleted.
        id_a = _inspect()
        _archive(id_a, paths)
        _force_archived_at(id_a, ten_days_ago)

        # B: verified (pending incorporation), archived long ago -> protected.
        id_b = _inspect()
        _archive(id_b, paths)
        _force_archived_at(id_b, ten_days_ago)
        label_b = {"verdict": "approved", "defect_classes": [], "boxes": []}
        store.verify(id_b, status="verified_correct", label=label_b, by="qa-reviewer")

        # C: verified AND already incorporated, archived long ago -> deleted (protection lifted).
        id_c = _inspect()
        _archive(id_c, paths)
        _force_archived_at(id_c, ten_days_ago)
        store.verify(id_c, status="verified_correct", label=label_b, by="qa-reviewer")
        store.mark_incorporated([id_c])

        # D: verified but archived only just now -> too new, left alone regardless.
        id_d = _inspect()
        _archive(id_d, paths)
        store.verify(id_d, status="verified_correct", label=label_b, by="qa-reviewer")

        # === 1: archived verified record still returned by the training query ===
        print("=== 1: list_verified_for_training() has no hidden lifecycle filter ===")
        verified_ids = {r.id for r in store.list_verified_for_training(COMPONENT_NAME)}
        ok1 = {id_b, id_c, id_d} <= verified_ids
        print(f"  verified ids returned (includes archived ones): {sorted(verified_ids & {id_a, id_b, id_c, id_d})}")
        print(f"-> {'PASS' if ok1 else 'FAIL'}: archived verified records (b, c, d) still come back.")
        print()

        # === 2, 3, 4: run retention cleanup, check who survives ==================
        print("=== 2/3/4: retention protects pending-verified, deletes incorporated + plain-unverified, spares too-new ===")
        result = retention.cleanup_archived_inspections(COMPONENT_NAME, registry=registry, now=now)
        print(f"  result: deleted={result.deleted} protected_pending_verified={result.protected_pending_verified} errors={result.errors}")

        a_after, b_after, c_after, d_after = store.get(id_a), store.get(id_b), store.get(id_c), store.get(id_d)
        ok2 = a_after is None  # plain unverified, old -> deleted
        ok3 = b_after is not None and b_after.verified_incorporation_status == "pending"  # protected
        ok4 = c_after is None  # verified but incorporated, old -> deleted (protection lifted)
        ok5 = d_after is not None  # too new -> spared regardless of verification
        print(f"  A (unverified, old): {'deleted' if a_after is None else 'STILL PRESENT'} (expected deleted)")
        print(f"  B (verified, pending, old): {'STILL PRESENT' if b_after is not None else 'deleted'} (expected still present)")
        print(f"  C (verified, incorporated, old): {'deleted' if c_after is None else 'STILL PRESENT'} (expected deleted)")
        print(f"  D (verified, too new): {'STILL PRESENT' if d_after is not None else 'deleted'} (expected still present)")
        ok234 = ok2 and ok3 and ok4 and ok5
        print(f"-> {'PASS' if ok234 else 'FAIL'}")
        print()

        # === 5: pending_verified_counts() surfaces the protected record ==========
        print("=== 5: pending_verified_counts() surfaces protected records for visibility ===")
        counts = retention.pending_verified_counts(registry=registry)
        ok6 = counts.get(COMPONENT_NAME, 0) >= 1
        print(f"  pending_verified_counts(): {counts}")
        print(f"-> {'PASS' if ok6 else 'FAIL'}: the still-pending record (b) is visible in the summary.")
        print()

        all_pass = all([ok1, ok234, ok6])
        print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED — see above'}")
    finally:
        registry.update_settings(COMPONENT_NAME, inspection_retention_days=old_retention_days)
        for i in created_ids:
            record = store.get(i)
            if record is None:
                continue
            for rel_path in (record.image_path, record.report_path):
                if rel_path:
                    (paths.root / rel_path).unlink(missing_ok=True)
        with connect(DB_PATH) as conn:
            placeholders = ", ".join("?" * len(created_ids))
            conn.execute(f"DELETE FROM {INSPECTIONS_TABLE} WHERE id IN ({placeholders})", created_ids)


if __name__ == "__main__":
    main()
