"""Fine-tunes a YOLO detector on this component's annotated image pool.

The pool (paths.yolo_images_dir / yolo_labels_dir / yolo_classes_file) is
filled in by onboarding via any of three annotation paths (pre-made YOLO
labels, mask-to-box conversion, manual drawing in the app) — this trainer
doesn't know or care which one produced it; they all converge on the same
images-plus-labels-on-disk format before training ever runs.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np
import yaml

from emil_ml.config.registry import Component
from emil_ml.core.base import BaseTrainer, EvaluationResult, TrainResult
from emil_ml.core.detection.yolo.annotation import read_classes_file, read_yolo_label
from emil_ml.core.detection.yolo.model import load_yolo_model
from emil_ml.core.evaluation import io as eval_io
from emil_ml.utils.paths import for_component

VAL_FRACTION = 0.2
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Baseline geometric/color-jitter values at yolo_augmentation_strength=1.0 —
# a moderate profile, scaled linearly down to 0 (fully disabled) by the
# strength setting. fliplr is deliberately not in here; see
# DEFAULT_YOLO_AUGMENTATION_STRENGTH's docstring for why.
_AUGMENTATION_BASELINE: dict[str, float] = {
    "degrees": 10.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 2.0,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
}


def _stratified_split(has_boxes: list[bool], rng: np.random.Generator) -> tuple[list[int], list[int]]:
    """Split image indices so both positive (has boxes) and negative examples land in train and val."""
    has_boxes_arr = np.array(has_boxes)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for flag in (True, False):
        group = np.where(has_boxes_arr == flag)[0].copy()
        if len(group) == 0:
            continue
        rng.shuffle(group)
        n_val = min(int(round(len(group) * VAL_FRACTION)), len(group) - 1) if len(group) > 1 else 0
        val_idx.extend(group[:n_val].tolist())
        train_idx.extend(group[n_val:].tolist())
    train_arr, val_arr = np.array(train_idx), np.array(val_idx)
    rng.shuffle(train_arr)
    rng.shuffle(val_arr)
    return train_arr.tolist(), val_arr.tolist()


class YoloTrainer(BaseTrainer):
    """Fine-tunes a pretrained YOLO model on approved-annotated images + boxes."""

    def train(self, component: Component) -> TrainResult:
        paths = for_component(component.name)

        image_paths = (
            sorted(
                p
                for p in paths.yolo_images_dir.iterdir()
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
            )
            if paths.yolo_images_dir.exists()
            else []
        )
        if not image_paths:
            raise ValueError(
                f"No annotated images found in {paths.yolo_images_dir} — onboard this component "
                "with at least one annotated image (via any of the three annotation methods) "
                "before training."
            )

        class_names = read_classes_file(paths.yolo_classes_file)
        if not class_names:
            raise ValueError(
                f"No classes defined at {paths.yolo_classes_file} — every YOLO component needs "
                "at least one class name, set during onboarding."
            )

        has_boxes = [
            len(read_yolo_label(paths.yolo_labels_dir / f"{p.stem}.txt")) > 0 for p in image_paths
        ]
        if not any(has_boxes):
            raise ValueError(
                f"None of the {len(image_paths)} images in {paths.yolo_labels_dir} have any "
                "annotated boxes — every label is empty. Training would produce a model that "
                "never detects anything. Go back to onboarding and draw/convert boxes for at "
                "least one image before training (a component's annotation pool is meant to "
                "hold a mix of positive and negative examples, not all-negative)."
            )

        rng = np.random.default_rng(0)
        train_idx, val_idx = _stratified_split(has_boxes, rng)
        if not val_idx:
            # Too little data to hold anything out — train and "validate" on
            # the same images rather than crash; Ultralytics still reports
            # metrics, just not on truly held-out data.
            val_idx = train_idx

        # Rebuild the scratch train/val split fresh every run, so a changed
        # annotation pool or class list never leaves stale files behind.
        if paths.yolo_dataset_dir.exists():
            shutil.rmtree(paths.yolo_dataset_dir)
        for split_name, indices in (("train", train_idx), ("val", val_idx)):
            images_dir = paths.yolo_dataset_dir / "images" / split_name
            labels_dir = paths.yolo_dataset_dir / "labels" / split_name
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            for i in indices:
                img_path = image_paths[i]
                shutil.copy2(img_path, images_dir / img_path.name)
                label_src = paths.yolo_labels_dir / f"{img_path.stem}.txt"
                label_dst = labels_dir / f"{img_path.stem}.txt"
                if label_src.exists():
                    shutil.copy2(label_src, label_dst)
                else:
                    label_dst.write_text("", encoding="utf-8")  # negative example

        data_yaml_path = paths.yolo_dataset_dir / "data.yaml"
        data_yaml_path.write_text(
            yaml.safe_dump(
                {
                    "path": str(paths.yolo_dataset_dir.resolve()),
                    "train": "images/train",
                    "val": "images/val",
                    "names": {i: name for i, name in enumerate(class_names)},
                }
            ),
            encoding="utf-8",
        )

        model = load_yolo_model(component.yolo_model_variant)
        augmentation_kwargs = {
            key: value * component.yolo_augmentation_strength
            for key, value in _AUGMENTATION_BASELINE.items()
        }
        # patience=0 means "disabled" to Ultralytics too, matching our own
        # early_stopping_patience convention exactly — no translation needed.
        #
        # amp=False: Ultralytics' automatic-mixed-precision startup check
        # exercises a Triton-JIT-compiled kernel, which segfaults (not a
        # catchable Python exception — a raw SIGSEGV, "segfault ... in
        # libtriton.so") on GPUs newer than Triton's precompiled-kernel
        # support — confirmed on an RTX 5070 (Blackwell, compute capability
        # 12.0a). Disabling AMP avoids that code path entirely; the cost is
        # somewhat slower GPU training, which is a much better trade than a
        # crash. Not made configurable since there's no upside to leaving it
        # on anywhere training actually runs today (CPU on Windows, where
        # AMP is moot; this specific GPU on WSL2, where it crashes).
        model.train(
            data=str(data_yaml_path),
            epochs=component.epochs,
            imgsz=component.image_size,
            batch=component.batch_size,
            patience=component.early_stopping_patience,
            mosaic=component.yolo_mosaic,
            cls=component.yolo_class_loss_weight,
            optimizer=component.yolo_optimizer,
            lr0=component.yolo_learning_rate,
            **augmentation_kwargs,
            project=str(paths.yolo_dataset_dir),
            name="run",
            exist_ok=True,
            seed=0,
            verbose=False,
            # Ultralytics' own per-epoch curves (results.png/results.csv)
            # and validation plots (confusion matrix, PR/F1/P/R curves) —
            # generated for free as part of this call, only when the
            # component actually wants an evaluation report (see
            # evaluate() below, which copies them out of
            # yolo_dataset_dir/run/ into the permanent evaluation/
            # directory before the next retrain wipes this scratch dir).
            # Off by default's speed benefit (see settings.py's
            # DEFAULT_GENERATE_EVALUATION_REPORT comment) is exactly this
            # flag staying False. bool(...): the Component field is a
            # SQLite INTEGER (0/1) under the hood, and Ultralytics' own
            # config validation rejects a plain int here — it strictly
            # requires an actual bool (confirmed: "'plots=1' is of invalid
            # type int").
            plots=bool(component.generate_evaluation_report),
            amp=False,
        )

        best_weights = paths.yolo_dataset_dir / "run" / "weights" / "best.pt"
        if not best_weights.exists():
            raise RuntimeError(f"YOLO training did not produce weights at {best_weights}")
        paths.models_dir.mkdir(parents=True, exist_ok=True)
        # Keep the model this run is about to replace, so a retrain that
        # turns out worse (e.g. a newly added class was under-annotated and
        # dragged accuracy down) can be manually recovered from
        # models/best.pt.bak instead of the previous model being gone the
        # instant training succeeds. Single rolling backup, not a full
        # history — this is a safety net for the one-run-back case, not a
        # version store.
        if paths.yolo_model_path.exists():
            shutil.copy2(paths.yolo_model_path, paths.yolo_model_backup_path)
        shutil.copy2(best_weights, paths.yolo_model_path)

        details: dict = {
            "yolo_model_variant": component.yolo_model_variant,
            "decision_rule": component.decision_rule,
            "yolo_mosaic": component.yolo_mosaic,
            "yolo_class_loss_weight": component.yolo_class_loss_weight,
            "yolo_augmentation_strength": component.yolo_augmentation_strength,
            "yolo_optimizer": component.yolo_optimizer,
            "yolo_learning_rate": component.yolo_learning_rate,
            "classes": class_names,
            "train_image_count": len(train_idx),
            "val_image_count": len(val_idx),
            "train_positive_count": sum(1 for i in train_idx if has_boxes[i]),
            "train_negative_count": sum(1 for i in train_idx if not has_boxes[i]),
        }

        # Best-effort: surface whatever validation metrics Ultralytics
        # computed (exact keys vary by version) without failing training if
        # the metrics object's shape doesn't match what we expect.
        try:
            metrics_source = model.trainer.validator.metrics.results_dict
            details["metrics"] = {
                k: float(v) for k, v in metrics_source.items() if isinstance(v, (int, float))
            }
        except Exception as exc:  # noqa: BLE001 - metrics are supplementary, never fatal
            details["metrics_note"] = f"Could not extract validation metrics: {exc}"

        # Per-class breakdown, not just the aggregate above — the whole point
        # of checking metrics after a class-extension retrain is "did the
        # new class actually learn something, and did the old ones regress",
        # neither of which the aggregate numbers can answer on their own.
        # ap_class_index only lists classes that had >=1 validation instance
        # this run; a class with none (e.g. a brand-new one whose only
        # annotated example landed in train, not val, on a small pool) is
        # filled in explicitly below rather than silently missing from the
        # dict, since "no data" and "0.0 recall" mean very different things.
        try:
            box_metrics = model.trainer.validator.metrics.box
            per_class_metrics: dict[str, dict[str, float | str]] = {}
            for i, class_id in enumerate(box_metrics.ap_class_index):
                class_id = int(class_id)
                name = class_names[class_id] if class_id < len(class_names) else str(class_id)
                precision, recall, ap50, ap = box_metrics.class_result(i)
                per_class_metrics[name] = {
                    "precision": float(precision),
                    "recall": float(recall),
                    "map50": float(ap50),
                    "map50_95": float(ap),
                }
            for name in class_names:
                if name not in per_class_metrics:
                    per_class_metrics[name] = {"note": "no validation instances for this class"}
            details["per_class_metrics"] = per_class_metrics
        except Exception as exc:  # noqa: BLE001 - metrics are supplementary, never fatal
            details["per_class_metrics_note"] = f"Could not extract per-class metrics: {exc}"

        return TrainResult(
            model_path=paths.yolo_model_path, threshold=DEFAULT_CONFIDENCE_THRESHOLD, details=details
        )

    def evaluate(self, component: Component, train_result: TrainResult, *, output_dir: Path) -> EvaluationResult:
        """Captures whatever Ultralytics ALREADY generated during train()
        (see train()'s own `plots=component.generate_evaluation_report`)
        rather than re-implementing YOLO's own plotting: per-class
        precision/recall/mAP@0.5/mAP@0.5:0.95 (already in
        `train_result.details["per_class_metrics"]`), a confusion matrix,
        PR/F1/P/R curves, and per-epoch loss + mAP curves (results.png/
        results.csv) — all copied out of the scratch `yolo_dataset_dir/run/`
        (rebuilt fresh on the NEXT retrain — see utils/paths.py) into the
        permanent, timestamped `output_dir` before that happens.

        Deliberately NO "accuracy over epochs" curve: YOLO doesn't train on
        or track accuracy at all — its own per-epoch progression is loss
        (box/cls/dfl) and mAP, both already in results.csv, both plotted
        here. Generating an accuracy curve for a method that has no
        accuracy would be exactly the "meaningless artifact forced on
        every method" this whole evaluation system exists to avoid (see
        core/base.py's BaseTrainer.evaluate()).
        """
        paths = for_component(component.name)
        run_dir = paths.yolo_dataset_dir / "run"
        plot_files: list[str] = []
        notes: list[str] = []

        if not run_dir.exists():
            notes.append(
                "No Ultralytics run directory found (either generate_evaluation_report was off during "
                "training, so plots=False was passed to model.train(), or yolo_dataset_dir was already "
                "cleaned up by a later retrain) — only the summary metrics captured during training itself "
                "are available."
            )
        else:
            # Curated set: evaluation-relevant plots, not the training-batch
            # preview images (train_batchN.jpg etc.) Ultralytics also writes.
            for name in (
                "results.png", "confusion_matrix.png", "confusion_matrix_normalized.png",
                "F1_curve.png", "PR_curve.png", "P_curve.png", "R_curve.png",
            ):
                src = run_dir / name
                if src.exists():
                    shutil.copy2(src, output_dir / name)
                    plot_files.append(name)

            results_csv = run_dir / "results.csv"
            if results_csv.exists():
                shutil.copy2(results_csv, output_dir / "results.csv")
                self._plot_epoch_curves_from_csv(results_csv, output_dir, plot_files)
            else:
                notes.append("Ultralytics did not produce results.csv for this run — per-epoch curves unavailable.")

        metrics = {
            "metrics": train_result.details.get("metrics", {}),
            "per_class_metrics": train_result.details.get("per_class_metrics", {}),
        }
        eval_io.write_metrics_json({**metrics, "notes": notes}, output_dir / "metrics.json")
        return EvaluationResult(artifacts_dir=output_dir, metrics=metrics, plot_files=plot_files, notes=notes)

    @staticmethod
    def _plot_epoch_curves_from_csv(results_csv: Path, output_dir: Path, plot_files: list[str]) -> None:
        """Ultralytics' results.csv has one row per epoch and columns like
        "train/box_loss", "val/box_loss", "metrics/mAP50(B)",
        "metrics/mAP50-95(B)" — matched by substring rather than an exact
        name, since the "(B)"-style task suffix has shifted across
        Ultralytics versions; substring matching is robust to that without
        needing to track the exact column name per version.
        """
        from emil_ml.core.evaluation import plots as eval_plots

        with open(results_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return
        columns = {name.strip(): [float(row[name]) for row in rows] for name in rows[0]}

        loss_series = {name: values for name, values in columns.items() if "loss" in name.lower()}
        if loss_series:
            eval_plots.plot_curves(loss_series, output_dir / "loss_curve.png", title="Loss over epochs", ylabel="Loss")
            plot_files.append("loss_curve.png")

        map_series = {name: values for name, values in columns.items() if "map" in name.lower()}
        if map_series:
            eval_plots.plot_curves(map_series, output_dir / "map_curve.png", title="mAP over epochs", ylabel="mAP")
            plot_files.append("map_curve.png")
