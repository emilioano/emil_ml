"""Scores an image with a trained PatchCore model against a threshold.

The threshold comparison always happens in anomalib's *normalized* score
space (0-1, decision boundary at 0.5) — see adapter.py's module docstring
for why that's the right space to compare in rather than the raw score.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from emil_ml.config.registry import Component
from emil_ml.core.anomaly.patchcore import adapter
from emil_ml.core.base import BasePredictor, PredictionResult
from emil_ml.utils.paths import for_component

_MODEL_CACHE: dict[tuple[str, float], Any] = {}


def _load_cached(model_path: Path):  # noqa: ANN201 - anomalib.models.Patchcore, imported lazily
    """Cache loaded models by (path, mtime) so retraining invalidates the cache."""
    mtime = model_path.stat().st_mtime
    key = (str(model_path), mtime)
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = adapter.load_model(model_path)
        _MODEL_CACHE[key] = model
    return model


class PatchCorePredictor(BasePredictor):
    """Loads the component's PatchCore model and scores images by patch-level anomaly distance."""

    def __init__(self, component: Component) -> None:
        super().__init__(component)
        if not component.model_path:
            raise ValueError(f"Component {component.name!r} has no trained model recorded")
        paths = for_component(component.name)
        model_path = paths.resolve_model_path(component.model_path)
        if not model_path.exists():
            raise ValueError(f"No trained PatchCore model found for component {component.name!r} at {model_path}")
        self._model = _load_cached(model_path)

    def predict(self, prepared_input: Image.Image) -> PredictionResult:
        if self.component.anomaly_threshold is None:
            raise ValueError(f"Component {self.component.name!r} has no anomaly threshold set")

        outcome = adapter.predict(self._model, prepared_input)
        threshold = self.component.anomaly_threshold
        verdict = "approved" if outcome.normalized_score <= threshold else "failed"
        return PredictionResult(
            verdict=verdict,
            score=outcome.normalized_score,
            threshold=threshold,
            details={"heatmap": outcome.heatmap},
        )
