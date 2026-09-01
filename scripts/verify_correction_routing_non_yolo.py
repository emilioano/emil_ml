"""Verifies the generalized (non-YOLO) correction routing added by
incorporate_verified_corrections(): classifier (binary, trainable on both
approved and failed) and an unsupervised method (autoencoder — never
trains on failed/defect images, confirmed by reading its trainer).

Deliberately bypasses orchestrator.run_inspection() / a real trained
predictor for these throwaway records — routing only reads
verified_label/image_path/verified_status off the InspectionRecord, so a
directly store.create()'d record with a real image file on disk is a
faster, equally valid way to exercise it than running detection for real.

1. classifier: a false-positive correction (approved label) routes into
   training/approved/; a false-negative correction (failed label) routes
   into training/failed/ — both directly trainable for this model_type,
   both traceably named `correction_<id>_copy<n>.<ext>`.
2. autoencoder (unsupervised): a false-positive correction (approved
   label) routes into training/approved/, same as classifier. A
   false-negative correction (failed label) is SKIPPED — never dumped
   into training/failed/ as if it were trainable defect data — logged
   clearly, and the record stays 'pending' (not force-incorporated).
3. incorporated really means "copied into training/": the files this
   script finds in training/approved/ and training/failed/ are read
   through the ordinary component paths, not a special test hook.

Run with: python scripts/verify_correction_routing_non_yolo.py
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
from emil_ml.core.inspections import store
from emil_ml.training import onboard
from emil_ml.utils.paths import for_component

CLASSIFIER_COMPONENT_DISPLAY_NAME = "Non-YOLO Routing Test Classifier"
UNSUPERVISED_COMPONENT_NAME = "tandborste"  # existing autoencoder component, reused as elsewhere in this session


def _make_image_file(paths, name: str) -> str:
    rng = np.random.default_rng(0)
    arr = np.clip(np.full((32, 32, 3), 120, dtype=np.int16) + rng.integers(-10, 10, (32, 32, 3)), 0, 255).astype("uint8")
    dest = paths.analyzed_approved_dir
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / name
    PILImage.fromarray(arr).save(path)
    return path.relative_to(paths.root).as_posix()


def _fabricate_record(component_name: str, paths, verdict: str, filename: str) -> int:
    """A throwaway InspectionRecord backed by a real image file, without
    running actual detection — see module docstring for why that's fine
    for a routing-only test."""
    image_rel_path = _make_image_file(paths, filename)
    record = store.create(
        component_name, verdict=verdict, score=0.5, threshold=0.5, image_path=image_rel_path, run_by="routing-test"
    )
    return record.id


def _cleanup(component_name: str, created_ids: list[int]) -> None:
    paths = for_component(component_name)
    for i in created_ids:
        record = store.get(i)
        if record is None:
            continue
        if record.image_path:
            (paths.root / record.image_path).unlink(missing_ok=True)
    if created_ids:
        with connect(DB_PATH) as conn:
            placeholders = ", ".join("?" * len(created_ids))
            conn.execute(f"DELETE FROM {INSPECTIONS_TABLE} WHERE id IN ({placeholders})", created_ids)


def main() -> None:
    configure_logging()
    registry = ComponentRegistry()

    all_pass = True

    # === 1: classifier — both approved and failed corrections are usable ===
    print("=== 1: classifier routes false_positive -> training/approved/, false_negative -> training/failed/ ===")
    classifier_name = None
    created_ids: list[int] = []
    try:
        component = onboard.create_component(
            CLASSIFIER_COMPONENT_DISPLAY_NAME, model_type="classifier", registry=registry
        )
        classifier_name = component.name
        paths = for_component(classifier_name)

        fp_id = _fabricate_record(classifier_name, paths, "failed", "fp_source.png")
        created_ids.append(fp_id)
        fp_label = {"verdict": "approved", "defect_classes": [], "boxes": []}
        store.verify(fp_id, status="verified_incorrect", label=fp_label, by="qa-reviewer")

        fn_id = _fabricate_record(classifier_name, paths, "approved", "fn_source.png")
        created_ids.append(fn_id)
        fn_label = {"verdict": "failed", "defect_classes": [], "boxes": []}
        store.verify(fn_id, status="verified_incorrect", label=fn_label, by="qa-reviewer")

        pending_ids = [r.id for r in onboard.list_pending_verified_corrections(classifier_name)]
        result = onboard.incorporate_verified_corrections(classifier_name, pending_ids, registry=registry)
        print(f"  result: {result}")

        approved_files = [p.name for p in paths.training_approved_dir.iterdir() if p.name.startswith("correction_")]
        failed_files = [p.name for p in paths.training_failed_dir.iterdir() if p.name.startswith("correction_")]
        print(f"  training/approved/ correction files: {approved_files}")
        print(f"  training/failed/ correction files: {failed_files}")

        ok1 = (
            result.incorporated == 2
            and result.per_bucket_counts.get("approved", 0) >= 1
            and result.per_bucket_counts.get("failed", 0) >= 1
            and any(f"correction_{fp_id}_" in f for f in approved_files)
            and any(f"correction_{fn_id}_" in f for f in failed_files)
            and store.get(fp_id).verified_incorporation_status == "incorporated"
            and store.get(fn_id).verified_incorporation_status == "incorporated"
        )
        print(f"-> {'PASS' if ok1 else 'FAIL'}")
        all_pass &= ok1
    finally:
        if classifier_name:
            _cleanup(classifier_name, created_ids)
            registry.delete(classifier_name)
            shutil.rmtree(for_component(classifier_name).root, ignore_errors=True)
    print()

    # === 2: unsupervised — false negative is skipped, not misrouted =========
    print("=== 2: autoencoder (unsupervised) — false_positive routes to approved/, false_negative is skipped, not dumped into failed/ ===")
    created_ids = []
    try:
        paths = for_component(UNSUPERVISED_COMPONENT_NAME)
        pre_approved_count = len(list(paths.training_approved_dir.iterdir()))
        pre_failed_count = len(list(paths.training_failed_dir.iterdir())) if paths.training_failed_dir.exists() else 0

        fp_id = _fabricate_record(UNSUPERVISED_COMPONENT_NAME, paths, "failed", "unsup_fp_source.png")
        created_ids.append(fp_id)
        store.verify(fp_id, status="verified_incorrect", label={"verdict": "approved", "defect_classes": [], "boxes": []}, by="qa-reviewer")

        fn_id = _fabricate_record(UNSUPERVISED_COMPONENT_NAME, paths, "approved", "unsup_fn_source.png")
        created_ids.append(fn_id)
        store.verify(fn_id, status="verified_incorrect", label={"verdict": "failed", "defect_classes": [], "boxes": []}, by="qa-reviewer")

        result2 = onboard.incorporate_verified_corrections(UNSUPERVISED_COMPONENT_NAME, [fp_id, fn_id], registry=registry)
        print(f"  result: {result2}")

        post_approved_count = len(list(paths.training_approved_dir.iterdir()))
        post_failed_count = len(list(paths.training_failed_dir.iterdir())) if paths.training_failed_dir.exists() else 0
        print(f"  training/approved/ count: {pre_approved_count} -> {post_approved_count}")
        print(f"  training/failed/ count: {pre_failed_count} -> {post_failed_count}")

        fp_after, fn_after = store.get(fp_id), store.get(fn_id)
        ok2 = (
            result2.incorporated == 1
            and result2.skipped_unusable == 1
            and post_approved_count > pre_approved_count
            and post_failed_count == pre_failed_count  # nothing dumped into failed/
            and fp_after.verified_incorporation_status == "incorporated"
            and fn_after.verified_incorporation_status == "pending"  # never force-incorporated
        )
        print(f"  fp incorporation status: {fp_after.verified_incorporation_status} (expected incorporated)")
        print(f"  fn incorporation status: {fn_after.verified_incorporation_status} (expected pending — gracefully skipped)")
        print(f"-> {'PASS' if ok2 else 'FAIL'}")
        all_pass &= ok2
    finally:
        _cleanup(UNSUPERVISED_COMPONENT_NAME, created_ids)
        paths = for_component(UNSUPERVISED_COMPONENT_NAME)
        for f in paths.training_approved_dir.iterdir():
            if f.name.startswith("correction_"):
                f.unlink(missing_ok=True)
    print()

    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED — see above'}")


if __name__ == "__main__":
    main()
