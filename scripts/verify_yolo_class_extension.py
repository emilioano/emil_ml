"""Verifies the YOLO class-extension feature end to end, on a disposable
throwaway component (deleted in `finally`, same convention as the other
verify_*.py scripts):

1. A new class appended via onboard.add_yolo_classes() gets the next free
   index; existing classes keep their original indices.
2. Adding a duplicate class name (case-insensitive) is rejected.
3. A label file saved BEFORE the class was added is untouched afterward —
   its class index still means what it always meant.
4. A new annotation using the new class's index is saved correctly.
5. Retraining regenerates data.yaml with the full, correct class list.
6. Retraining reports per-class metrics (both classes present in the dict).
7. Retraining a second time backs up the previous best.pt to
   models/best.bak.pt before overwriting it — verified by exact byte
   comparison against what best.pt held right before the second run.

Run with: python scripts/verify_yolo_class_extension.py
(CPU is fine; epochs/image_size are kept small so both training runs are fast.)
"""

from __future__ import annotations

import shutil
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from PIL import Image as PILImage

from emil_ml.config.logging_config import configure_logging
from emil_ml.config.registry import ComponentRegistry
from emil_ml.training import onboard
from emil_ml.utils.paths import for_component

COMPONENT_DISPLAY_NAME = "Yolo Class Extension Test"


def _make_image(rng: np.random.Generator, seed_offset: int) -> PILImage.Image:
    arr = np.clip(
        np.full((64, 64, 3), 120, dtype=np.int16) + rng.integers(-10, 10, (64, 64, 3)), 0, 255
    ).astype("uint8")
    return PILImage.fromarray(arr)


def main() -> None:
    configure_logging()
    registry = ComponentRegistry()
    rng = np.random.default_rng(0)
    name = None
    try:
        component = onboard.create_yolo_component(
            COMPONENT_DISPLAY_NAME,
            class_names=["scratch"],
            registry=registry,
            image_size=64,
            epochs=1,
            batch_size=2,
            early_stopping_patience=0,
        )
        name = component.name
        paths = for_component(name)

        # === Seed the pool: several "scratch" (class 0) annotations =========
        print("=== Seeding pool with class 0 ('scratch') annotations ===")
        for i in range(6):
            img = _make_image(rng, i)
            buf = img.tobytes()
            import io

            byte_io = io.BytesIO()
            img.save(byte_io, format="PNG")
            onboard.add_yolo_annotation(
                name, f"scratch_{i}.png", byte_io.getvalue(), [(0, 0.5, 0.5, 0.2, 0.2)]
            )
        pre_add_label = (paths.yolo_labels_dir / "scratch_0.txt").read_text(encoding="utf-8")
        print(f"  pre-add label content for scratch_0.txt: {pre_add_label!r}")
        print()

        # === 1: append a new class ===========================================
        print("=== 1: onboard.add_yolo_classes() appends, preserving existing indices ===")
        classes_before = onboard.get_yolo_classes(name)
        updated = onboard.add_yolo_classes(name, ["dent"])
        classes_after = onboard.get_yolo_classes(name)
        print(f"  before: {classes_before}")
        print(f"  after:  {classes_after}")
        ok1 = classes_before == ["scratch"] and classes_after == ["scratch", "dent"] and updated == classes_after
        print(f"-> {'PASS' if ok1 else 'FAIL'}: 'scratch' stayed index 0, 'dent' appended at index 1.")
        print()

        # === 2: duplicate class name rejected ================================
        print("=== 2: duplicate class name (case-insensitive) is rejected ===")
        ok2 = False
        try:
            onboard.add_yolo_classes(name, ["Scratch"])
        except ValueError as exc:
            ok2 = True
            print(f"  raised ValueError as expected: {exc}")
        print(f"-> {'PASS' if ok2 else 'FAIL'}: duplicate class name raises ValueError, list unchanged.")
        print(f"  classes still: {onboard.get_yolo_classes(name)}")
        print()

        # === 3: pre-existing label file untouched by the class addition =====
        print("=== 3: a label saved BEFORE the class was added still reads the same after ===")
        post_add_label = (paths.yolo_labels_dir / "scratch_0.txt").read_text(encoding="utf-8")
        ok3 = post_add_label == pre_add_label and post_add_label.strip().startswith("0 ")
        print(f"  post-add label content for scratch_0.txt: {post_add_label!r}")
        print(f"-> {'PASS' if ok3 else 'FAIL'}: unchanged, still class index 0.")
        print()

        # === 4: new annotation using the new class's index ===================
        print("=== 4: new annotation using class 1 ('dent') saves correctly ===")
        for i in range(6):
            img = _make_image(rng, 100 + i)
            import io

            byte_io = io.BytesIO()
            img.save(byte_io, format="PNG")
            onboard.add_yolo_annotation(
                name, f"dent_{i}.png", byte_io.getvalue(), [(1, 0.5, 0.5, 0.3, 0.3)]
            )
        dent_label = (paths.yolo_labels_dir / "dent_0.txt").read_text(encoding="utf-8")
        ok4 = dent_label.strip().startswith("1 ")
        print(f"  dent_0.txt content: {dent_label!r}")
        print(f"-> {'PASS' if ok4 else 'FAIL'}: saved with class index 1.")
        print()

        # === 5 & 6: retrain #1, check data.yaml + per-class metrics =========
        print("=== 5 & 6: first retrain -> data.yaml has both classes, per-class metrics present ===")
        result1 = onboard.train_component(name, registry=registry)
        data_yaml_text = (paths.yolo_dataset_dir / "data.yaml").read_text(encoding="utf-8")
        print("  data.yaml:")
        print(data_yaml_text)
        ok5 = "0: scratch" in data_yaml_text and "1: dent" in data_yaml_text
        print(f"-> {'PASS' if ok5 else 'FAIL'}: data.yaml names cover both classes at their correct indices.")

        per_class = result1.details.get("per_class_metrics")
        print(f"  per_class_metrics: {per_class}")
        ok6 = isinstance(per_class, dict) and "scratch" in per_class and "dent" in per_class
        print(f"-> {'PASS' if ok6 else 'FAIL'}: per-class metrics reported for both classes.")
        print()

        # === 7: second retrain backs up the previous best.pt =================
        print("=== 7: second retrain backs up the pre-retrain best.pt before overwriting ===")
        pre_second_bytes = paths.yolo_model_path.read_bytes()
        backup_existed_before = paths.yolo_model_backup_path.exists()
        onboard.train_component(name, registry=registry)
        backup_bytes = paths.yolo_model_backup_path.read_bytes() if paths.yolo_model_backup_path.exists() else None
        ok7 = (
            not backup_existed_before
            and backup_bytes is not None
            and backup_bytes == pre_second_bytes
        )
        print(f"  backup existed before 2nd retrain: {backup_existed_before} (expected False)")
        print(f"  backup matches pre-retrain best.pt: {backup_bytes == pre_second_bytes if backup_bytes else 'N/A'}")
        print(f"-> {'PASS' if ok7 else 'FAIL'}: previous model backed up to models/best.bak.pt exactly.")
        print()

        all_pass = all([ok1, ok2, ok3, ok4, ok5, ok6, ok7])
        print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED — see above'}")
    finally:
        if name:
            registry.delete(name)
            shutil.rmtree(for_component(name).root, ignore_errors=True)


if __name__ == "__main__":
    main()
