"""Verifies the correction feedback loop end to end, on a disposable
throwaway YOLO component (deleted in `finally`, same convention as the
other verify_*.py scripts). Covers exactly the four chains called out
explicitly for this feature:

1. Flag a YOLO wrong-class case -> (simulated) annotate the correct
   box/class -> store.verify() is the only thing that writes the
   verified_* columns -> the row ends up verified_incorrect with
   verified_error_type='wrong_class' and the explicit corrected label,
   prediction fields (verdict/score/defect_classes) untouched.
2. A hidden approved record (approved_handling='hide_from_default_view',
   the dangerous false-negative case) -> flagged as false negative ->
   annotated -> verify() -> verified_incorrect/false_negative. The
   station's own "Show approved" toggle reachability is covered
   separately by verify_inspection_station.py's live AppTest check; this
   script confirms the actual correction plumbing works once revealed.
3. Confirming a prediction as correct -> verify() with verified_correct
   and an explicit, non-empty label (never an implicit "same as
   prediction" shortcut).
4. Retraining consumption: list_verified_for_training() is the ONLY
   source incorporate_verified_corrections() reads from (never the
   archive folder or inspections table directly) — unverified records,
   including an auto-acknowledged approved one, are never included;
   consumed examples get marked 'incorporated' so a second consumption
   pass never double-adds them; a verified_correct record with no box
   data is skipped (stays 'pending') rather than wrongly materialized;
   per-class counts are reported.

Run with: python scripts/verify_correction_loop.py
"""

from __future__ import annotations

import io
import shutil
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from PIL import Image as PILImage

from emil_ml.config.database import connect
from emil_ml.config.logging_config import configure_logging
from emil_ml.config.registry import ComponentRegistry
from emil_ml.config.settings import DB_PATH, INSPECTIONS_TABLE
from emil_ml.core.inspections import orchestrator, store
from emil_ml.training import onboard
from emil_ml.utils.paths import for_component

COMPONENT_DISPLAY_NAME = "Verify Loop Test"


def _make_image(rng: np.random.Generator) -> bytes:
    arr = np.clip(
        np.full((64, 64, 3), 120, dtype=np.int16) + rng.integers(-10, 10, (64, 64, 3)), 0, 255
    ).astype("uint8")
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _delete_inspections_for(component_name: str) -> None:
    """Test-only cleanup: store.py deliberately has no generic "delete
    every inspection for a component" function (inspection history is
    never supposed to disappear in normal operation) — but a disposable
    test component reusing the same slug across runs needs its OWN
    leftover rows gone, or a later run's list_verified_for_training()
    query would see a previous run's orphaned rows too (registry.delete()
    only removes the component row, not inspections referencing its name)."""
    with connect(DB_PATH) as conn:
        conn.execute(f"DELETE FROM {INSPECTIONS_TABLE} WHERE component_name = ?", (component_name,))


def main() -> None:
    configure_logging()
    registry = ComponentRegistry()
    rng = np.random.default_rng(0)
    name = None
    try:
        _delete_inspections_for("verify-loop-test")  # leftover rows from a prior interrupted run, if any
        component = onboard.create_yolo_component(
            COMPONENT_DISPLAY_NAME,
            class_names=["scratch", "dent"],
            registry=registry,
            image_size=64,
            epochs=1,
            batch_size=2,
            early_stopping_patience=0,
            reporting_enabled=False,
        )
        name = component.name
        paths = for_component(name)

        print("=== Setup: seeding pool + training an initial model ===")
        for i in range(6):
            onboard.add_yolo_annotation(name, f"scratch_{i}.png", _make_image(rng), [(0, 0.5, 0.5, 0.2, 0.2)])
        for i in range(4):
            onboard.add_yolo_annotation(name, f"dent_{i}.png", _make_image(rng), [(1, 0.5, 0.5, 0.3, 0.3)])
        onboard.train_component(name, registry=registry)
        class_names = onboard.get_yolo_classes(name)
        sample_image_path = str(paths.yolo_images_dir / "scratch_0.png")
        print(f"  classes: {class_names}")
        print()

        # === Setup: 5 real inspections covering every case this loop must handle ===
        record_a, _ = orchestrator.run_inspection(
            sample_image_path, name, registry=registry, async_report=False, run_by="tester", threshold_override=-1.0
        )
        original_a_verdict, original_a_classes = record_a.verdict, list(record_a.defect_classes)

        registry.update_settings(name, approved_handling="hide_from_default_view")
        record_b, _ = orchestrator.run_inspection(
            sample_image_path, name, registry=registry, async_report=False, run_by="tester", threshold_override=999.0
        )
        original_b_verdict, original_b_classes = record_b.verdict, list(record_b.defect_classes)

        record_c, _ = orchestrator.run_inspection(
            sample_image_path, name, registry=registry, async_report=False, run_by="tester", threshold_override=-1.0
        )

        registry.update_settings(name, approved_handling="auto_acknowledge")
        record_d, _ = orchestrator.run_inspection(
            sample_image_path, name, registry=registry, async_report=False, run_by="tester", threshold_override=999.0
        )
        registry.update_settings(name, approved_handling="hide_from_default_view")

        record_e, _ = orchestrator.run_inspection(
            sample_image_path, name, registry=registry, async_report=False, run_by="tester", threshold_override=-1.0
        )

        # === Chain 1: wrong-class correction, via verify() only ==============
        print("=== 1: flag wrong-class -> annotate -> verify() -> verified_incorrect + label, prediction untouched ===")
        corrected_class_a = "dent" if "dent" not in original_a_classes else "scratch"
        boxes_a = [(class_names.index(corrected_class_a), 0.5, 0.5, 0.25, 0.25)]
        label_a = onboard.build_verified_label("failed", boxes_a, class_names)
        store.verify(record_a.id, status="verified_incorrect", label=label_a, by="qa-reviewer")
        reloaded_a = store.get(record_a.id)
        ok1 = (
            reloaded_a.verdict == original_a_verdict
            and reloaded_a.defect_classes == original_a_classes
            and reloaded_a.verified_status == "verified_incorrect"
            and reloaded_a.verified_error_type == "wrong_class"
            and reloaded_a.verified_label == label_a
        )
        print(f"  original prediction: verdict={original_a_verdict} defect_classes={original_a_classes}")
        print(f"  after correction: verified_status={reloaded_a.verified_status} error_type={reloaded_a.verified_error_type}")
        print(f"  prediction still: verdict={reloaded_a.verdict} defect_classes={reloaded_a.defect_classes}")
        print(f"-> {'PASS' if ok1 else 'FAIL'}")
        print()

        # === Chain 2: hidden approved record -> false negative -> annotated -> verified ===
        print("=== 2: hidden approved record flagged as false negative -> annotated -> verify() ===")
        boxes_b = [(class_names.index("scratch"), 0.4, 0.4, 0.2, 0.2)]
        label_b = onboard.build_verified_label("failed", boxes_b, class_names)
        store.verify(record_b.id, status="verified_incorrect", label=label_b, by="qa-reviewer")
        reloaded_b = store.get(record_b.id)
        ok2 = (
            reloaded_b.verdict == original_b_verdict == "approved"
            and reloaded_b.defect_classes == original_b_classes
            and reloaded_b.verified_status == "verified_incorrect"
            and reloaded_b.verified_error_type == "false_negative"
            and reloaded_b.verified_label == label_b
        )
        print(f"  original prediction: verdict={original_b_verdict} (approved_handling was hide_from_default_view)")
        print(f"  after correction: verified_status={reloaded_b.verified_status} error_type={reloaded_b.verified_error_type}")
        print(f"-> {'PASS' if ok2 else 'FAIL'}: the most dangerous error type (missed real defect) reaches verify() correctly.")
        print()

        # === Chain 3: verify as correct — light action, explicit non-empty label ===
        print("=== 3: verify a prediction as correct -> verify() with verified_correct + explicit label ===")
        label_c = {"verdict": record_c.verdict, "defect_classes": record_c.defect_classes, "boxes": []}
        store.verify(record_c.id, status="verified_correct", label=label_c, by="qa-reviewer")
        reloaded_c = store.get(record_c.id)
        ok3 = (
            reloaded_c.verified_status == "verified_correct"
            and reloaded_c.verified_error_type is None
            and reloaded_c.verified_label == label_c
            and bool(reloaded_c.verified_label)
        )
        print(f"  verified_status={reloaded_c.verified_status} error_type={reloaded_c.verified_error_type} label={reloaded_c.verified_label}")
        print(f"-> {'PASS' if ok3 else 'FAIL'}")
        print()

        # === Chain 4: retraining consumes ONLY via list_verified_for_training() ===
        print("=== 4: retraining consumption — verified-only, pending->incorporated, per-class report ===")
        verified_ids_before = {r.id for r in store.list_verified_for_training(name)}
        print(f"  list_verified_for_training() returns: {sorted(verified_ids_before)}")
        ok4a = (
            {record_a.id, record_b.id, record_c.id} <= verified_ids_before
            and record_d.id not in verified_ids_before
            and record_e.id not in verified_ids_before
        )
        print(f"-> {'PASS' if ok4a else 'FAIL'}: only the 3 verified records are visible; auto-acked approved (d) and plain unverified (e) are not.")

        pre_pool_count = len(list(paths.yolo_images_dir.iterdir()))
        pending_ids = [r.id for r in onboard.list_pending_verified_corrections(name)]
        incorporation = onboard.incorporate_verified_corrections(name, pending_ids, registry=registry)
        post_pool_count = len(list(paths.yolo_images_dir.iterdir()))
        print(f"  incorporation result: {incorporation}")
        print(f"  pool image count: {pre_pool_count} -> {post_pool_count}")

        reloaded_a2, reloaded_b2, reloaded_c2 = store.get(record_a.id), store.get(record_b.id), store.get(record_c.id)
        ok4b = (
            incorporation.incorporated == 2  # a (wrong_class) + b (false_negative) — both have real boxes
            and incorporation.skipped_unusable == 1  # c: verified_correct, verdict=failed, no boxes -> unusable
            and reloaded_a2.verified_incorporation_status == "incorporated"
            and reloaded_b2.verified_incorporation_status == "incorporated"
            and reloaded_c2.verified_incorporation_status == "pending"
            and post_pool_count > pre_pool_count
        )
        print(
            f"  incorporation status: a={reloaded_a2.verified_incorporation_status} "
            f"b={reloaded_b2.verified_incorporation_status} c={reloaded_c2.verified_incorporation_status}"
        )
        print(f"-> {'PASS' if ok4b else 'FAIL'}: only usable corrections consumed and marked incorporated; unusable one stays pending.")

        ok4c = (
            incorporation.per_bucket_counts.get(corrected_class_a, 0) >= 2
            and incorporation.per_bucket_counts.get("scratch", 0) >= 2
        )
        print(f"  per_bucket_counts: {incorporation.per_bucket_counts}")
        print(f"-> {'PASS' if ok4c else 'FAIL'}: corrected classes show up in the per-class report, oversampled (weighted heavier).")

        # Second incorporation pass: must not double-add the already-incorporated corrections.
        pending_ids_2 = [r.id for r in onboard.list_pending_verified_corrections(name)]
        second_incorporation = onboard.incorporate_verified_corrections(name, pending_ids_2, registry=registry)
        ok4d = second_incorporation.incorporated == 0
        print(f"  second incorporation pass result: {second_incorporation}")
        print(f"-> {'PASS' if ok4d else 'FAIL'}: already-incorporated corrections are never re-added on a later retrain.")

        ok4 = ok4a and ok4b and ok4c and ok4d
        print()

        all_pass = all([ok1, ok2, ok3, ok4])
        print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED — see above'}")
    finally:
        if name:
            _delete_inspections_for(name)
            registry.delete(name)
            shutil.rmtree(for_component(name).root, ignore_errors=True)


if __name__ == "__main__":
    main()
