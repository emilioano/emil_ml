"""Trains a supervised CNN classifier on both approved and failed images.

Unlike the autoencoder (unsupervised, approved-only), this needs labeled
examples of both classes: approved -> 0, failed -> 1. With the small
datasets this is meant for (tens of images per class), it relies on transfer
learning (see model.py) plus class weighting and data augmentation to avoid
overfitting, and reports validation metrics — especially recall on the
failed class, since missing a real defect is worse than a false alarm.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tensorflow import keras

from emil_ml.config.registry import Component
from emil_ml.config.settings import VALID_CLASS_WEIGHT_STRATEGIES
from emil_ml.core.base import BaseTrainer, EvaluationResult, TrainResult
from emil_ml.core.classification.cnn_classifier.model import build_classifier
from emil_ml.core.evaluation import io as eval_io
from emil_ml.core.evaluation import plots
from emil_ml.utils import image_io
from emil_ml.utils.paths import for_component

VAL_FRACTION = 0.2
DEFAULT_DECISION_THRESHOLD = 0.5

APPROVED_LABEL = 0
FAILED_LABEL = 1


def _stratified_split(
    labels: np.ndarray, val_fraction: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices so both classes are represented in train and val where possible.

    A class with only one example goes entirely to train (there's nothing
    sensible to hold out); otherwise at least one example of that class
    stays in train even if val_fraction would round it away.
    """
    train_idx: list[int] = []
    val_idx: list[int] = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0].copy()
        rng.shuffle(cls_idx)
        if len(cls_idx) > 1:
            n_val = min(int(round(len(cls_idx) * val_fraction)), len(cls_idx) - 1)
        else:
            n_val = 0
        val_idx.extend(cls_idx[:n_val].tolist())
        train_idx.extend(cls_idx[n_val:].tolist())
    train_arr = np.array(train_idx, dtype=int)
    val_arr = np.array(val_idx, dtype=int)
    rng.shuffle(train_arr)
    rng.shuffle(val_arr)
    return train_arr, val_arr


def _class_weight(train_labels: np.ndarray, strategy: str) -> dict[int, float]:
    """"balanced": weight each class inversely to its frequency. "none": equal weight."""
    if strategy not in VALID_CLASS_WEIGHT_STRATEGIES:
        raise ValueError(
            f"Unknown class_weight_strategy {strategy!r}; must be one of {VALID_CLASS_WEIGHT_STRATEGIES}"
        )
    classes = np.unique(train_labels)
    if strategy == "none":
        return {int(c): 1.0 for c in classes}
    total = len(train_labels)
    return {
        int(cls): total / (len(classes) * int(np.sum(train_labels == cls))) for cls in classes
    }


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Precision/recall/confusion matrix with the failed class (1) as positive.

    Also reports balanced accuracy — mean of recall on each class — since
    plain accuracy and plain recall_failed can both look deceptively good for
    a degenerate "always predict one class" model under class imbalance.
    """
    tp = int(np.sum((y_pred == FAILED_LABEL) & (y_true == FAILED_LABEL)))
    fp = int(np.sum((y_pred == FAILED_LABEL) & (y_true == APPROVED_LABEL)))
    fn = int(np.sum((y_pred == APPROVED_LABEL) & (y_true == FAILED_LABEL)))
    tn = int(np.sum((y_pred == APPROVED_LABEL) & (y_true == APPROVED_LABEL)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_failed = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    recall_approved = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    return {
        "val_accuracy": accuracy,
        "val_precision_failed": precision,
        "val_recall_failed": recall_failed,
        "val_balanced_accuracy": (recall_failed + recall_approved) / 2,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def _early_stopping_callbacks(patience: int, has_val: bool) -> list:
    """EarlyStopping callback list, or [] if patience <= 0 (disabled).

    Monitors val_loss when a validation split exists, else falls back to
    training loss (there's nothing else to monitor without one).
    """
    if patience <= 0:
        return []
    monitor = "val_loss" if has_val else "loss"
    return [keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True)]


def _evaluate(model: keras.Model, X_val: np.ndarray, y_val: np.ndarray) -> dict:
    val_probs = model.predict(X_val, verbose=0).ravel()
    val_preds = (val_probs >= DEFAULT_DECISION_THRESHOLD).astype(int)
    metrics = _binary_metrics(y_val.astype(int), val_preds)
    metrics["val_probability_stats"] = {
        "min": float(np.min(val_probs)),
        "max": float(np.max(val_probs)),
        "mean": float(np.mean(val_probs)),
    }
    return metrics


class ClassifierTrainer(BaseTrainer):
    """Supervised: trains on approved (label 0) AND failed (label 1) images."""

    def train(self, component: Component) -> TrainResult:
        # Weight init and the augmentation layers are stochastic by default,
        # so identical data could otherwise give different results between
        # runs — seeded for reproducibility (and so head-vs-fine-tune
        # comparisons below reflect the data, not initialization luck).
        keras.utils.set_random_seed(0)

        paths = for_component(component.name)
        approved_images = image_io.load_dataset(paths.training_approved_dir, component.image_size)
        failed_images = image_io.load_dataset(paths.training_failed_dir, component.image_size)

        if len(approved_images) == 0:
            raise ValueError(f"No approved training images found in {paths.training_approved_dir}")
        if len(failed_images) == 0:
            raise ValueError(
                f"No failed training images found in {paths.training_failed_dir} — "
                "the classifier needs labeled examples of both classes to train on."
            )

        images = np.concatenate([approved_images, failed_images], axis=0)
        labels = np.concatenate(
            [
                np.full(len(approved_images), APPROVED_LABEL, dtype=np.float32),
                np.full(len(failed_images), FAILED_LABEL, dtype=np.float32),
            ]
        )

        rng = np.random.default_rng(0)
        train_idx, val_idx = _stratified_split(labels, VAL_FRACTION, rng)
        X_train, y_train = images[train_idx], labels[train_idx]
        X_val, y_val = images[val_idx], labels[val_idx]
        has_val = len(X_val) > 0

        class_weight = _class_weight(y_train, component.class_weight_strategy)

        model = build_classifier(
            image_size=component.image_size,
            base_model_name=component.base_model,
            pooling=component.pooling,
            augmentation_strength=component.augmentation_strength,
        )
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        fit_batch_size = min(component.batch_size, len(X_train))
        validation_data = (X_val, y_val) if has_val else None
        early_stopping = _early_stopping_callbacks(component.early_stopping_patience, has_val)
        history_head = model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=component.epochs,
            batch_size=fit_batch_size,
            class_weight=class_weight,
            shuffle=True,
            verbose=0,
            callbacks=early_stopping,
        )

        head_metrics = _evaluate(model, X_val, y_val) if has_val else None
        head_weights = model.get_weights()

        # Fine-tuning phase: unfreeze the base's top layers, drop the learning
        # rate, keep training. Not guaranteed to help on tiny datasets — it
        # can just as easily destabilize an already-good decision boundary
        # (confirmed by testing) — so it's evaluated below, never trusted
        # blindly: whichever phase validates better (by balanced accuracy,
        # not plain recall — recall alone is trivially maxed out by a
        # degenerate "always predict failed" model) is what gets saved.
        # Guard against `layers[:-0]`, which is `layers[:0]` (empty) in Python,
        # not "keep everything" — that would paradoxically unfreeze the whole
        # base when the user asks to unfreeze zero layers.
        model.base.trainable = component.fine_tune_unfreeze_layers > 0
        if component.fine_tune_unfreeze_layers > 0:
            for layer in model.base.layers[: -component.fine_tune_unfreeze_layers]:
                layer.trainable = False
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=component.fine_tune_learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        history_fine_tune = model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=component.fine_tune_epochs,
            batch_size=fit_batch_size,
            class_weight=class_weight,
            shuffle=True,
            verbose=0,
            # Fresh callback instance — don't reuse the head phase's, even
            # though Keras resets its internal state on_train_begin anyway.
            callbacks=_early_stopping_callbacks(component.early_stopping_patience, has_val),
        )

        details: dict = {
            "base_model": component.base_model,
            "pooling": component.pooling,
            "class_weight_strategy": component.class_weight_strategy,
            "augmentation_strength": component.augmentation_strength,
            "fine_tune_epochs": component.fine_tune_epochs,
            "fine_tune_learning_rate": component.fine_tune_learning_rate,
            "fine_tune_unfreeze_layers": component.fine_tune_unfreeze_layers,
            "early_stopping_patience": component.early_stopping_patience,
            "head_epochs_run": len(history_head.epoch),
            "fine_tune_epochs_run": len(history_fine_tune.epoch),
            "train_approved_count": int(np.sum(y_train == APPROVED_LABEL)),
            "train_failed_count": int(np.sum(y_train == FAILED_LABEL)),
            "val_approved_count": int(np.sum(y_val == APPROVED_LABEL)),
            "val_failed_count": int(np.sum(y_val == FAILED_LABEL)),
            "class_weight": {str(k): v for k, v in class_weight.items()},
            "history": {"head": history_head.history, "fine_tune": history_fine_tune.history},
        }

        if has_val:
            fine_tune_metrics = _evaluate(model, X_val, y_val)
            if fine_tune_metrics["val_balanced_accuracy"] >= head_metrics["val_balanced_accuracy"]:
                selected_phase, selected_metrics = "fine_tune", fine_tune_metrics
            else:
                selected_phase, selected_metrics = "head", head_metrics
                model.set_weights(head_weights)  # fine-tuning made it worse; roll back

            details["selected_phase"] = selected_phase
            details["head_metrics"] = head_metrics
            details["fine_tune_metrics"] = fine_tune_metrics
            details.update(selected_metrics)
            # Nested "metrics" key, same convention YOLO/PatchCore/Isolation
            # Forest use — lets grid_search.py rank trials for any
            # model_type the same generic way. Only the flat float fields —
            # confusion_matrix/val_probability_stats are nested dicts, not
            # something a single metric_key lookup can compare across trials.
            details["metrics"] = {
                k: v for k, v in selected_metrics.items() if isinstance(v, (int, float))
            }
        else:
            details["selected_phase"] = "fine_tune"
            details["val_note"] = (
                "Not enough data for a validation split, so the fine-tuned model was kept "
                "without being able to confirm it's actually better than the head-only model."
            )

        paths.models_dir.mkdir(parents=True, exist_ok=True)
        model.save(paths.classifier_model_path)

        return TrainResult(
            model_path=paths.classifier_model_path, threshold=DEFAULT_DECISION_THRESHOLD, details=details
        )

    def evaluate(self, component: Component, train_result: TrainResult, *, output_dir: Path) -> EvaluationResult:
        """Classification report + confusion matrix (sklearn, so this
        generalizes to a future multi-class classifier — not hand-derived
        from a 2x2 confusion matrix the way this trainer's own
        `_binary_metrics()` is) on the SAME validation split train() used
        (reconstructed here via the identical seeded `_stratified_split()`
        call, rather than threading raw arrays through TrainResult.details
        — that would mean serializing whole image arrays into a JSON blob),
        plus loss/accuracy curves across both the head and fine-tune
        phases (from `train_result.details["history"]`, already captured
        by train() — no re-training needed to plot it).
        """
        from sklearn.metrics import classification_report, confusion_matrix

        paths = for_component(component.name)
        approved_images = image_io.load_dataset(paths.training_approved_dir, component.image_size)
        failed_images = image_io.load_dataset(paths.training_failed_dir, component.image_size)
        images = np.concatenate([approved_images, failed_images], axis=0)
        labels = np.concatenate(
            [
                np.full(len(approved_images), APPROVED_LABEL, dtype=np.float32),
                np.full(len(failed_images), FAILED_LABEL, dtype=np.float32),
            ]
        )
        rng = np.random.default_rng(0)
        _, val_idx = _stratified_split(labels, VAL_FRACTION, rng)

        history = train_result.details.get("history", {})
        head_history = history.get("head", {})
        fine_tune_history = history.get("fine_tune", {})
        phase_boundary = len(head_history.get("loss", []))

        plot_files: list[str] = []
        plots.plot_curves(
            {"loss": head_history.get("loss", []) + fine_tune_history.get("loss", []),
             "val_loss": head_history.get("val_loss", []) + fine_tune_history.get("val_loss", [])},
            output_dir / "loss_curve.png", title="Loss over epochs", ylabel="Loss", phase_boundary=phase_boundary,
        )
        plot_files.append("loss_curve.png")
        plots.plot_curves(
            {"accuracy": head_history.get("accuracy", []) + fine_tune_history.get("accuracy", []),
             "val_accuracy": head_history.get("val_accuracy", []) + fine_tune_history.get("val_accuracy", [])},
            output_dir / "accuracy_curve.png", title="Accuracy over epochs", ylabel="Accuracy", phase_boundary=phase_boundary,
        )
        plot_files.append("accuracy_curve.png")

        metrics: dict = {}
        notes: list[str] = []
        if len(val_idx) == 0:
            notes.append(
                "No validation split was available (too little data) — classification report and "
                "confusion matrix require held-out labeled examples. Only the loss/accuracy curves "
                "were generated."
            )
        else:
            X_val, y_val = images[val_idx], labels[val_idx].astype(int)
            model = keras.models.load_model(train_result.model_path)
            probs = model.predict(X_val, verbose=0).ravel()
            y_pred = (probs >= DEFAULT_DECISION_THRESHOLD).astype(int)

            class_names = ["approved", "failed"]
            report = classification_report(
                y_val, y_pred, labels=[APPROVED_LABEL, FAILED_LABEL], target_names=class_names,
                output_dict=True, zero_division=0,
            )
            cm = confusion_matrix(y_val, y_pred, labels=[APPROVED_LABEL, FAILED_LABEL])
            plots.plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png")
            plot_files.append("confusion_matrix.png")

            metrics["classification_report"] = report
            metrics["confusion_matrix"] = cm.tolist()
            metrics["class_names"] = class_names

        eval_io.write_metrics_json({"metrics": metrics, "notes": notes}, output_dir / "metrics.json")
        return EvaluationResult(artifacts_dir=output_dir, metrics=metrics, plot_files=plot_files, notes=notes)
