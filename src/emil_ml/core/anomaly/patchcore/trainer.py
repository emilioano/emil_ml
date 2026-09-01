"""Trains PatchCore on approved images only, via anomalib (see adapter.py).

Unsupervised, like the autoencoder: the anomaly threshold is derived
automatically from the training run's own score calibration; any failed
images are used only to validate/refine that threshold and to compute
image-level AUROC, never to train on.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from emil_ml.config.registry import Component
from emil_ml.core.anomaly.patchcore import adapter
from emil_ml.core.base import BaseTrainer, EvaluationResult, TrainResult
from emil_ml.core.evaluation import io as eval_io
from emil_ml.core.evaluation.unsupervised import evaluate_anomaly_scores
from emil_ml.utils.paths import for_component


def _has_images(directory) -> bool:  # noqa: ANN001
    return directory.exists() and any(directory.iterdir())


class PatchCoreTrainer(BaseTrainer):
    """Unsupervised, patch-level: trains only on approved images via anomalib's PatchCore."""

    def train(self, component: Component) -> TrainResult:
        paths = for_component(component.name)
        if not _has_images(paths.training_approved_dir):
            raise ValueError(f"No approved training images found in {paths.training_approved_dir}")

        abnormal_dir = paths.training_failed_dir if _has_images(paths.training_failed_dir) else None

        outcome = adapter.train(
            normal_dir=paths.training_approved_dir,
            abnormal_dir=abnormal_dir,
            results_dir=paths.patchcore_results_dir,
            backbone=component.patchcore_backbone,
            coreset_sampling_ratio=component.patchcore_coreset_sampling_ratio,
            num_neighbors=component.patchcore_num_neighbors,
            batch_size=component.batch_size,
        )

        paths.models_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(outcome.checkpoint_path, paths.patchcore_model_path)
        # The scratch Lightning results dir (logs, intermediate checkpoints)
        # isn't needed once the checkpoint we care about is copied out —
        # same "rebuilt fresh, not kept" spirit as YOLO's yolo_dataset_dir.
        shutil.rmtree(paths.patchcore_results_dir, ignore_errors=True)

        details: dict = {
            "patchcore_backbone": component.patchcore_backbone,
            "patchcore_coreset_sampling_ratio": component.patchcore_coreset_sampling_ratio,
            "patchcore_num_neighbors": component.patchcore_num_neighbors,
            "raw_threshold": outcome.raw_threshold,
            "used_failed_images_for_calibration": abnormal_dir is not None,
        }
        if outcome.metrics:
            details["metrics"] = outcome.metrics

        return TrainResult(
            model_path=paths.patchcore_model_path, threshold=adapter.NORMALIZED_THRESHOLD, details=details
        )

    def evaluate(self, component: Component, train_result: TrainResult, *, output_dir: Path) -> EvaluationResult:
        """The shared score-histogram/ROC/PR/threshold-metrics procedure
        only (core/evaluation/unsupervised.py) — no loss curve, same
        reasoning as Isolation Forest's evaluate(): PatchCore's "training"
        is a single coreset-extraction pass, not iterative loss
        minimization over epochs.

        Unlike autoencoder/Isolation Forest, adapter.train() doesn't
        expose raw per-image scores (only anomalib's own aggregate test
        metrics, already in `train_result.details["metrics"]`) — so this
        re-runs inference on the same approved/failed images via the
        already-trained checkpoint, through the exact same adapter.predict()
        every real inspection uses (see predictor.py), rather than reaching
        into anomalib's Lightning internals to extract what train() already
        computed once.
        """
        from PIL import Image as PILImage

        paths = for_component(component.name)
        model = adapter.load_model(train_result.model_path)

        def _scores(directory: Path) -> list[float]:
            if not _has_images(directory):
                return []
            scores = []
            for image_path in sorted(directory.iterdir()):
                if not image_path.is_file():
                    continue
                try:
                    image = PILImage.open(image_path)
                except Exception:  # noqa: BLE001 - an unreadable file shouldn't abort the whole evaluation
                    continue
                scores.append(adapter.predict(model, image).normalized_score)
            return scores

        approved_scores = _scores(paths.training_approved_dir)
        failed_scores = _scores(paths.training_failed_dir)

        metrics, plot_files, notes = evaluate_anomaly_scores(
            approved_scores, failed_scores or None, train_result.threshold, output_dir, title_prefix="PatchCore ",
        )
        metrics.update(train_result.details.get("metrics", {}))  # anomalib's own image_AUROC/image_F1Score, if present

        eval_io.write_metrics_json({"metrics": metrics, "notes": notes}, output_dir / "metrics.json")
        return EvaluationResult(artifacts_dir=output_dir, metrics=metrics, plot_files=plot_files, notes=notes)
