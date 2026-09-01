"""Trains a convolutional autoencoder on approved images only.

The anomaly threshold is derived automatically from the reconstruction-error
distribution of the approved set; failed images (if any) are used only to
validate that threshold, never to train on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tensorflow import keras

from emil_ml.config.registry import Component
from emil_ml.config.settings import VALID_SCORE_METHODS
from emil_ml.core.anomaly.autoencoder.model import build_autoencoder
from emil_ml.core.base import BaseTrainer, EvaluationResult, TrainResult
from emil_ml.core.evaluation import io as eval_io
from emil_ml.core.evaluation import plots
from emil_ml.core.evaluation.unsupervised import evaluate_anomaly_scores
from emil_ml.utils import image_io
from emil_ml.utils.paths import for_component


def _early_stopping_callbacks(patience: int, monitor: str) -> list:
    """EarlyStopping callback list, or [] if patience <= 0 (disabled)."""
    if patience <= 0:
        return []
    return [keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True)]


def _reconstruction_errors(model, images: np.ndarray, score_method: str) -> np.ndarray:
    """Per-image score from a batch of images, using the component's score_method.

    "global_mean": average squared error over the whole image — the original
    approach. "local_max": the single worst-reconstructed pixel (averaged
    across channels first) — more sensitive to a small, localized defect that
    global_mean would dilute across an otherwise well-reconstructed image.
    """
    if len(images) == 0:
        return np.array([], dtype=np.float32)
    if score_method not in VALID_SCORE_METHODS:
        raise ValueError(f"Unknown score_method {score_method!r}; must be one of {VALID_SCORE_METHODS}")

    reconstructions = model.predict(images, verbose=0)
    squared_error = np.square(images - reconstructions)  # (N, H, W, C)

    if score_method == "global_mean":
        return np.mean(squared_error, axis=(1, 2, 3))
    # local_max
    per_pixel_error = np.mean(squared_error, axis=3)  # (N, H, W)
    return np.max(per_pixel_error, axis=(1, 2))


class AutoencoderTrainer(BaseTrainer):
    """Unsupervised: trains only on approved images, flags anomalies via reconstruction error."""

    def train(self, component: Component) -> TrainResult:
        paths = for_component(component.name)
        approved_images = image_io.load_dataset(paths.training_approved_dir, component.image_size)
        if len(approved_images) == 0:
            raise ValueError(f"No approved training images found in {paths.training_approved_dir}")

        model = build_autoencoder(image_size=component.image_size, latent_dim=component.latent_dim)
        model.compile(optimizer="adam", loss="mse")

        fit_batch_size = min(component.batch_size, len(approved_images))
        history = model.fit(
            approved_images,
            approved_images,
            epochs=component.epochs,
            batch_size=fit_batch_size,
            shuffle=True,
            verbose=0,
            # No validation split here (approved-only training), so this
            # monitors training loss rather than a held-out val_loss.
            callbacks=_early_stopping_callbacks(component.early_stopping_patience, monitor="loss"),
        )

        approved_errors = _reconstruction_errors(model, approved_images, component.score_method)
        threshold = float(np.percentile(approved_errors, component.threshold_percentile))

        failed_images = image_io.load_dataset(paths.training_failed_dir, component.image_size)
        failed_errors = (
            _reconstruction_errors(model, failed_images, component.score_method)
            if len(failed_images)
            else None
        )

        paths.models_dir.mkdir(parents=True, exist_ok=True)
        model.save(paths.model_path)

        details: dict = {
            "score_method": component.score_method,
            "threshold_percentile": component.threshold_percentile,
            "early_stopping_patience": component.early_stopping_patience,
            "epochs_run": len(history.epoch),
            "approved_errors": [float(e) for e in approved_errors],
            "approved_count": int(len(approved_errors)),
            "approved_mean_error": float(np.mean(approved_errors)),
            "approved_max_error": float(np.max(approved_errors)),
            "history": history.history,
        }
        if failed_errors is not None and len(failed_errors) > 0:
            correctly_flagged = int(np.sum(failed_errors > threshold))
            failed_flagged_ratio = correctly_flagged / len(failed_errors)
            details.update(
                {
                    "failed_errors": [float(e) for e in failed_errors],
                    "failed_count": int(len(failed_errors)),
                    "failed_mean_error": float(np.mean(failed_errors)),
                    "failed_correctly_flagged": correctly_flagged,
                    "failed_flagged_ratio": failed_flagged_ratio,
                    # Nested "metrics" key, same convention YOLO/PatchCore/
                    # Isolation Forest use — lets grid_search.py rank trials
                    # for any model_type the same generic way.
                    "metrics": {"failed_flagged_ratio": failed_flagged_ratio},
                }
            )

        return TrainResult(model_path=paths.model_path, threshold=threshold, details=details)

    def evaluate(self, component: Component, train_result: TrainResult, *, output_dir: Path) -> EvaluationResult:
        """Loss curve (the ONE unsupervised method that trains iteratively —
        PatchCore/Isolation Forest deliberately do not get this, see their
        own evaluate()) plus the shared score-histogram/ROC/PR/threshold-
        metrics procedure (core/evaluation/unsupervised.py) on the raw
        per-image reconstruction errors train() already computed and
        stored in `details["approved_errors"]`/`details["failed_errors"]`
        — no re-inference needed here.
        """
        details = train_result.details
        plot_files: list[str] = []

        history = details.get("history", {})
        if history.get("loss"):
            plots.plot_curves(
                {k: v for k, v in history.items() if "loss" in k},
                output_dir / "loss_curve.png", title="Loss over epochs", ylabel="Loss",
            )
            plot_files.append("loss_curve.png")

        metrics, score_plot_files, notes = evaluate_anomaly_scores(
            details.get("approved_errors", []), details.get("failed_errors"), train_result.threshold, output_dir,
        )
        plot_files.extend(score_plot_files)

        eval_io.write_metrics_json({"metrics": metrics, "notes": notes}, output_dir / "metrics.json")
        return EvaluationResult(artifacts_dir=output_dir, metrics=metrics, plot_files=plot_files, notes=notes)
