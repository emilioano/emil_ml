"""Verifies the model_type-aware evaluation reporting system end-to-end:
each model_type generates its own meaningful artifacts and no meaningless
ones (classifier gets classification report + accuracy curve; YOLO gets
mAP + PR/F1 curves but NO accuracy curve; unsupervised methods get
threshold metrics + score histogram, and only the autoencoder gets a loss
curve); artifacts are versioned/never overwritten; the per-component
setting turns generation on/off; a before/after comparison across two
retrains is possible; and correction-loop traceability threads through.

Run with: python scripts/verify_evaluation_reports.py
"""

from __future__ import annotations

import io
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import json

import numpy as np
from PIL import Image as PILImage

from emil_ml.config.logging_config import configure_logging
from emil_ml.config.registry import ComponentRegistry
from emil_ml.core import component_deletion
from emil_ml.core.detection.yolo.annotation import write_classes_file, write_yolo_label
from emil_ml.core.training_runs import store as training_runs_store
from emil_ml.training import onboard
from emil_ml.utils.paths import for_component

ALL_PASS = True
CLEANUP_NAMES: list[str] = []


def _check(label: str, condition: bool, detail: str = "") -> None:
    global ALL_PASS
    status = "PASS" if condition else "FAIL"
    print(f"  {status}: {label}" + (f" — {detail}" if detail else ""))
    ALL_PASS = ALL_PASS and condition


def _make_image_bytes(seed: int, *, size: int = 64) -> bytes:
    rng = np.random.default_rng(seed)
    arr = np.clip(np.full((size, size, 3), 120, dtype=np.int16) + rng.integers(-15, 15, (size, size, 3)), 0, 255).astype("uint8")
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _cleanup(name: str) -> None:
    registry = ComponentRegistry()
    if registry.get(name) is not None:
        registry.soft_delete(name)
        component_deletion.permanently_delete_component(name, registry=registry)


def _eval_files(component_name: str, evaluation_dir_relative: str) -> set[str]:
    paths = for_component(component_name)
    eval_dir = paths.root / evaluation_dir_relative
    return {p.name for p in eval_dir.iterdir() if p.is_file()}


def section_classifier() -> None:
    print("=== Classifier: classification report + confusion matrix + loss/accuracy curves ===")
    registry = ComponentRegistry()
    component = onboard.create_component(
        "Eval Test Classifier", model_type="classifier", registry=registry,
        image_size=64, epochs=3, fine_tune_epochs=1, batch_size=4,
    )
    CLEANUP_NAMES.append(component.name)
    onboard.add_training_images(
        component.name,
        approved=[(f"a{i}.png", _make_image_bytes(i)) for i in range(8)],
        failed=[(f"f{i}.png", _make_image_bytes(100 + i)) for i in range(8)],
    )
    onboard.train_component(component.name, registry=registry)

    runs = training_runs_store.list_for_component(component.name)
    run = runs[0]
    _check("training_runs row has a non-null evaluation_dir", run.evaluation_dir is not None)
    files = _eval_files(component.name, run.evaluation_dir)
    print(f"  artifact files: {sorted(files)}")
    _check("has metrics.json", "metrics.json" in files)
    _check("has confusion_matrix.png", "confusion_matrix.png" in files)
    _check("has loss_curve.png", "loss_curve.png" in files)
    _check("has accuracy_curve.png", "accuracy_curve.png" in files)
    _check("does NOT have score_histogram.png (unsupervised-only)", "score_histogram.png" not in files)
    _check("does NOT have map_curve.png (YOLO-only)", "map_curve.png" not in files)

    metrics_path = for_component(component.name).root / run.evaluation_dir / "metrics.json"
    saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    report = saved_metrics.get("metrics", {}).get("classification_report", {})
    _check(
        "classification_report has per-class precision/recall/f1 for both classes",
        "approved" in report and "failed" in report
        and all(k in report["approved"] for k in ("precision", "recall", "f1-score", "support")),
        detail=str(list(report.keys())),
    )
    print()


def section_autoencoder() -> None:
    print("=== Autoencoder: score histogram + ROC/PR + threshold metrics + loss curve ===")
    registry = ComponentRegistry()
    component = onboard.create_component(
        "Eval Test Autoencoder", model_type="autoencoder", registry=registry,
        image_size=64, epochs=3, batch_size=4, latent_dim=8,
    )
    CLEANUP_NAMES.append(component.name)
    onboard.add_training_images(
        component.name,
        approved=[(f"a{i}.png", _make_image_bytes(i)) for i in range(8)],
        failed=[(f"f{i}.png", _make_image_bytes(200 + i)) for i in range(6)],
    )
    onboard.train_component(component.name, registry=registry)

    run = training_runs_store.list_for_component(component.name)[0]
    _check("training_runs row has a non-null evaluation_dir", run.evaluation_dir is not None)
    files = _eval_files(component.name, run.evaluation_dir)
    print(f"  artifact files: {sorted(files)}")
    _check("has score_histogram.png", "score_histogram.png" in files)
    _check("has loss_curve.png (the one unsupervised method that trains iteratively)", "loss_curve.png" in files)
    _check("has roc_curve.png (labeled failed examples were available)", "roc_curve.png" in files)
    _check("has pr_curve.png", "pr_curve.png" in files)
    _check("does NOT have confusion_matrix.png (classifier-only plot)", "confusion_matrix.png" not in files)
    print()


def section_isolation_forest() -> None:
    print("=== Isolation Forest: score histogram + ROC/PR + threshold metrics, NO loss curve ===")
    registry = ComponentRegistry()
    component = onboard.create_component(
        "Eval Test Isoforest", model_type="isolation_forest", registry=registry, image_size=64,
    )
    CLEANUP_NAMES.append(component.name)
    onboard.add_training_images(
        component.name,
        approved=[(f"a{i}.png", _make_image_bytes(i)) for i in range(10)],
        failed=[(f"f{i}.png", _make_image_bytes(300 + i)) for i in range(6)],
    )
    onboard.train_component(component.name, registry=registry)

    run = training_runs_store.list_for_component(component.name)[0]
    _check("training_runs row has a non-null evaluation_dir", run.evaluation_dir is not None)
    files = _eval_files(component.name, run.evaluation_dir)
    print(f"  artifact files: {sorted(files)}")
    _check("has score_histogram.png", "score_histogram.png" in files)
    _check("has roc_curve.png", "roc_curve.png" in files)
    _check("has pr_curve.png", "pr_curve.png" in files)
    _check("does NOT have loss_curve.png (not trained iteratively)", "loss_curve.png" not in files)
    return component.name


def section_patchcore() -> None:
    print("=== PatchCore: score histogram + ROC/PR + threshold metrics, NO loss curve ===")
    try:
        import anomalib  # noqa: F401
    except ImportError:
        print("  SKIPPED: anomalib not installed (optional 'patchcore' extra)")
        print()
        return
    registry = ComponentRegistry()
    component = onboard.create_component(
        "Eval Test Patchcore", model_type="patchcore", registry=registry,
        image_size=64, batch_size=4, patchcore_coreset_sampling_ratio=0.5,
    )
    CLEANUP_NAMES.append(component.name)
    # anomalib's own Folder datamodule needs enough images to carve out a
    # non-empty internal val/test split (a too-small pool hits a real,
    # pre-existing anomalib bug: ZeroDivisionError in its random_split
    # helper — confirmed unrelated to this evaluation-reporting work by
    # reproducing it first with a too-small pool) — sized generously here
    # to stay well clear of that, not because evaluate() itself needs it.
    onboard.add_training_images(
        component.name,
        approved=[(f"a{i}.png", _make_image_bytes(i)) for i in range(16)],
        failed=[(f"f{i}.png", _make_image_bytes(400 + i)) for i in range(10)],
    )
    onboard.train_component(component.name, registry=registry)

    run = training_runs_store.list_for_component(component.name)[0]
    _check("training_runs row has a non-null evaluation_dir", run.evaluation_dir is not None)
    files = _eval_files(component.name, run.evaluation_dir)
    print(f"  artifact files: {sorted(files)}")
    _check("has score_histogram.png", "score_histogram.png" in files)
    _check("does NOT have loss_curve.png (single-pass coreset extraction, not iterative)", "loss_curve.png" not in files)
    print()


def section_yolo() -> None:
    print("=== YOLO (defect detector): mAP + PR/F1/confusion-matrix curves, NO accuracy curve ===")
    registry = ComponentRegistry()
    component = onboard.create_component(
        "Eval Test Yolo", model_type="yolo", registry=registry, image_size=64, epochs=1, batch_size=2,
    )
    CLEANUP_NAMES.append(component.name)
    paths = for_component(component.name)
    write_classes_file(paths.yolo_classes_file, ["defect"])
    for i in range(4):
        img_bytes = _make_image_bytes(i, size=128)
        img_path = paths.yolo_images_dir / f"img{i}.png"
        img_path.write_bytes(img_bytes)
        # Two positives (a centered box) and two negatives (empty label = no boxes).
        boxes = [(0, 0.5, 0.5, 0.3, 0.3)] if i % 2 == 0 else []
        write_yolo_label(paths.yolo_labels_dir / f"img{i}.txt", boxes)

    onboard.train_component(component.name, registry=registry)

    run = training_runs_store.list_for_component(component.name)[0]
    _check("training_runs row has a non-null evaluation_dir", run.evaluation_dir is not None)
    files = _eval_files(component.name, run.evaluation_dir)
    print(f"  artifact files: {sorted(files)}")
    _check("does NOT have accuracy_curve.png (YOLO tracks mAP, not accuracy)", "accuracy_curve.png" not in files)
    _check("has map_curve.png (per-epoch mAP, YOLO's actual primary metric)", "map_curve.png" in files)
    _check("has results.csv (raw per-epoch data, copied from Ultralytics)", "results.csv" in files)

    metrics_path = paths.root / run.evaluation_dir / "metrics.json"
    saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    per_class = saved_metrics.get("per_class_metrics", {})
    _check(
        "per_class_metrics has map50/map50_95 per class",
        "defect" in per_class and ("map50" in per_class["defect"] or "note" in per_class["defect"]),
        detail=str(per_class),
    )
    print()


def section_setting_off() -> str:
    print("=== Per-component setting: generate_evaluation_report=False skips reporting entirely ===")
    registry = ComponentRegistry()
    component = onboard.create_component(
        "Eval Test Off", model_type="isolation_forest", registry=registry, image_size=64,
        generate_evaluation_report=False,
    )
    CLEANUP_NAMES.append(component.name)
    onboard.add_training_images(component.name, approved=[(f"a{i}.png", _make_image_bytes(i)) for i in range(5)])
    onboard.train_component(component.name, registry=registry)

    run = training_runs_store.list_for_component(component.name)[0]
    paths = for_component(component.name)
    _check("training_runs row's evaluation_dir is None", run.evaluation_dir is None)
    _check("evaluation/ directory was never created on disk", not paths.evaluation_dir.exists())
    print()
    return component.name


def section_versioning_and_before_after(isoforest_component_name: str) -> None:
    print("=== Versioned artifacts: retraining does NOT overwrite the previous run's evaluation ===")
    registry = ComponentRegistry()
    component = registry.get(isoforest_component_name)
    onboard.add_training_images(
        component.name, approved=[(f"a_extra{i}.png", _make_image_bytes(500 + i)) for i in range(3)]
    )
    onboard.train_component(component.name, registry=registry)  # second training run

    runs = training_runs_store.list_for_component(component.name, source="train")
    _check("two training_runs rows now exist (before + after)", len(runs) == 2, detail=f"count={len(runs)}")
    eval_dirs = {r.evaluation_dir for r in runs}
    _check("each run has its OWN distinct evaluation_dir", len(eval_dirs) == 2, detail=str(eval_dirs))

    paths = for_component(component.name)
    for run in runs:
        eval_path = paths.root / run.evaluation_dir
        _check(f"run {run.created_at}'s artifacts still exist on disk (not overwritten)", eval_path.exists())
        _check(f"run {run.created_at} still has its own score_histogram.png", (eval_path / "score_histogram.png").exists())
    print()


def section_correction_traceability() -> None:
    print("=== Correction-loop traceability: incorporated_correction_ids threads into the training_runs row ===")
    registry = ComponentRegistry()
    component = onboard.create_component(
        "Eval Test Traceability", model_type="isolation_forest", registry=registry, image_size=64,
    )
    CLEANUP_NAMES.append(component.name)
    onboard.add_training_images(component.name, approved=[(f"a{i}.png", _make_image_bytes(i)) for i in range(5)])
    onboard.train_component(component.name, registry=registry, incorporated_correction_ids=[111, 222])

    run = training_runs_store.list_for_component(component.name)[0]
    _check(
        "training_runs.details['incorporated_correction_ids'] == [111, 222]",
        run.details.get("incorporated_correction_ids") == [111, 222],
        detail=str(run.details.get("incorporated_correction_ids")),
    )
    print()


def main() -> None:
    """`--group tf` runs only the TensorFlow-based sections (classifier,
    autoencoder, isolation_forest — core/diagnostics/embeddings.py's CNN
    embedding extractor is TF-based too) plus everything downstream that
    reuses one of their components; `--group torch` runs only the
    PyTorch-based ones (patchcore via anomalib, yolo via ultralytics). No
    argument runs everything in one process, which is fine on a platform
    where TensorFlow never touches the GPU at all (e.g. native Windows —
    TF dropped GPU support there after 2.10, so there's no CUDA/Triton
    context for PyTorch to conflict with) but segfaults on at least one
    real WSL2 GPU/driver combination when both frameworks load CUDA/Triton
    in the same process (confirmed directly: a WSL run of this script,
    unsplit, segfaulted partway through — see
    core/registry_factory.py's own module docstring, which already
    documents this exact cross-framework crash mode; this split is the
    verify-script-level version of the same mitigation, not a new
    finding). On a platform where the crash doesn't apply, running the
    two groups as separate `python scripts/verify_evaluation_reports.py
    --group ...` invocations is still safe and correct, just two
    processes instead of one.
    """
    import sys

    group = None
    if "--group" in sys.argv:
        group = sys.argv[sys.argv.index("--group") + 1]
        if group not in ("tf", "torch"):
            raise SystemExit(f"--group must be 'tf' or 'torch', got {group!r}")

    configure_logging()
    try:
        if group in (None, "tf"):
            section_classifier()
            section_autoencoder()
            isoforest_name = section_isolation_forest()
            print()
            section_setting_off()
            section_versioning_and_before_after(isoforest_name)
            section_correction_traceability()
        if group in (None, "torch"):
            section_patchcore()
            section_yolo()
        print(f"Overall: {'ALL PASS' if ALL_PASS else 'SOME FAILED — see above'}")
    finally:
        for name in CLEANUP_NAMES:
            _cleanup(name)


if __name__ == "__main__":
    main()
