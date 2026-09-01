"""Scores an image's CNN embedding with a trained Isolation Forest against a threshold."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from PIL import Image

from emil_ml.config.registry import Component
from emil_ml.core.base import BasePredictor, PredictionResult
from emil_ml.core.diagnostics import embeddings as diag_embeddings
from emil_ml.utils.paths import for_component

_MODEL_CACHE: dict[tuple[str, float], Any] = {}
_SCALER_CACHE: dict[tuple[str, float], Any] = {}


def _load_cached(cache: dict[tuple[str, float], Any], path: Path) -> Any:
    """Cache a joblib-loaded object by (path, mtime) so retraining invalidates the cache."""
    mtime = path.stat().st_mtime
    key = (str(path), mtime)
    obj = cache.get(key)
    if obj is None:
        obj = joblib.load(path)
        cache[key] = obj
    return obj


class IsolationForestPredictor(BasePredictor):
    """Loads the component's Isolation Forest (+ scaler, if trained with one) and scores images by CNN embedding."""

    def __init__(self, component: Component) -> None:
        super().__init__(component)
        if not component.model_path:
            raise ValueError(f"Component {component.name!r} has no trained model recorded")
        paths = for_component(component.name)
        model_path = paths.resolve_model_path(component.model_path)
        if not model_path.exists():
            raise ValueError(
                f"No trained Isolation Forest model found for component {component.name!r} at {model_path}"
            )
        self._model = _load_cached(_MODEL_CACHE, model_path)
        self._scaler = (
            _load_cached(_SCALER_CACHE, paths.isolation_forest_scaler_path)
            if paths.isolation_forest_scaler_path.exists()
            else None
        )

    def predict(self, prepared_input: Image.Image) -> PredictionResult:
        if self.component.anomaly_threshold is None:
            raise ValueError(f"Component {self.component.name!r} has no anomaly threshold set")

        embedding = diag_embeddings.embed_image(prepared_input).reshape(1, -1)
        if self._scaler is not None:
            embedding = self._scaler.transform(embedding)

        # score_samples() is LOWER for more anomalous points; negate to match
        # the trainer's threshold, which is in EMIL's higher-is-worse space.
        score = float(-self._model.score_samples(embedding)[0])
        threshold = self.component.anomaly_threshold
        verdict = "approved" if score <= threshold else "failed"
        return PredictionResult(verdict=verdict, score=score, threshold=threshold, details={})
