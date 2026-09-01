"""Trains an Isolation Forest on CNN embeddings of approved images only.

Unsupervised, like the autoencoder and PatchCore: the anomaly threshold is
derived automatically from the fitted forest's own score calibration
(`contamination`); any failed images are used only to validate that
threshold and compute image-level AUROC, never to train on.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from emil_ml.config.registry import Component
from emil_ml.core.anomaly.isolation_forest.model import build_isolation_forest
from emil_ml.core.base import BaseTrainer, EvaluationResult, TrainResult
from emil_ml.core.diagnostics import embeddings as diag_embeddings
from emil_ml.core.evaluation import io as eval_io
from emil_ml.core.evaluation.unsupervised import evaluate_anomaly_scores
from emil_ml.utils.paths import for_component


def _parse_contamination(raw: str) -> str | float:
    return "auto" if raw == "auto" else float(raw)


class IsolationForestTrainer(BaseTrainer):
    """Unsupervised: trains only on approved images, flags anomalies via an Isolation Forest on CNN embeddings."""

    def train(self, component: Component) -> TrainResult:
        paths = for_component(component.name)
        labeled = diag_embeddings.extract_embeddings(paths)
        if len(labeled.labels) == 0:
            raise ValueError(f"No training images found for component {component.name!r}")

        labels_arr = np.asarray(labeled.labels)
        approved_embeddings = labeled.embeddings[labels_arr == "approved"]
        if len(approved_embeddings) == 0:
            raise ValueError(f"No approved training images found in {paths.training_approved_dir}")

        scaler: StandardScaler | None = None
        train_embeddings = approved_embeddings
        if component.isolation_forest_standardize:
            scaler = StandardScaler()
            train_embeddings = scaler.fit_transform(approved_embeddings)

        model = build_isolation_forest(
            n_estimators=component.isolation_forest_n_estimators,
            contamination=_parse_contamination(component.isolation_forest_contamination),
            max_features=component.isolation_forest_max_features,
        )
        model.fit(train_embeddings)

        # score_samples() is LOWER for more anomalous points; EMIL's
        # app-wide convention is higher score = more anomalous, so every
        # score/threshold here is negated relative to sklearn's own sign.
        threshold = float(-model.offset_)

        failed_embeddings = labeled.embeddings[labels_arr == "failed"]

        paths.models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, paths.isolation_forest_model_path)
        if scaler is not None:
            joblib.dump(scaler, paths.isolation_forest_scaler_path)
        elif paths.isolation_forest_scaler_path.exists():
            # A prior training run left a scaler behind but this run has
            # standardize off — remove it so the predictor doesn't load a
            # stale scaler that no longer matches this model.
            paths.isolation_forest_scaler_path.unlink()

        # Computed unconditionally (cheap — train_embeddings is already in
        # memory) and stored raw, not just aggregated, so evaluate() can
        # build a score histogram/ROC/PR curve without re-running
        # inference — see core/evaluation/unsupervised.py, which every
        # unsupervised method's evaluate() feeds these same raw scores
        # into.
        approved_scores = -model.score_samples(train_embeddings)
        details: dict = {
            "isolation_forest_n_estimators": component.isolation_forest_n_estimators,
            "isolation_forest_contamination": component.isolation_forest_contamination,
            "isolation_forest_max_features": component.isolation_forest_max_features,
            "isolation_forest_standardize": bool(component.isolation_forest_standardize),
            "approved_count": int(len(approved_embeddings)),
            "approved_scores": [float(s) for s in approved_scores],
        }

        if len(failed_embeddings) > 0:
            eval_failed = scaler.transform(failed_embeddings) if scaler is not None else failed_embeddings
            failed_scores = -model.score_samples(eval_failed)
            y_true = np.concatenate([np.zeros(len(approved_scores)), np.ones(len(failed_scores))])
            y_score = np.concatenate([approved_scores, failed_scores])
            correctly_flagged = int(np.sum(failed_scores > threshold))
            details["failed_scores"] = [float(s) for s in failed_scores]
            details["metrics"] = {
                "failed_count": int(len(failed_scores)),
                "failed_correctly_flagged": correctly_flagged,
                "failed_flagged_ratio": correctly_flagged / len(failed_scores),
                "roc_auc": float(roc_auc_score(y_true, y_score)),
            }

        return TrainResult(model_path=paths.isolation_forest_model_path, threshold=threshold, details=details)

    def evaluate(self, component: Component, train_result: TrainResult, *, output_dir: Path) -> EvaluationResult:
        """The shared score-histogram/ROC/PR/threshold-metrics procedure
        only (core/evaluation/unsupervised.py) — deliberately NO loss
        curve: Isolation Forest fits in one shot (a forest of random
        splits), not iteratively over epochs, so there is no per-epoch
        loss to plot (see autoencoder/trainer.py's evaluate(), the one
        unsupervised method that does have one).
        """
        details = train_result.details
        metrics, plot_files, notes = evaluate_anomaly_scores(
            details.get("approved_scores", []), details.get("failed_scores"), train_result.threshold, output_dir,
            title_prefix="Isolation Forest ",
        )
        eval_io.write_metrics_json({"metrics": metrics, "notes": notes}, output_dir / "metrics.json")
        return EvaluationResult(artifacts_dir=output_dir, metrics=metrics, plot_files=plot_files, notes=notes)
