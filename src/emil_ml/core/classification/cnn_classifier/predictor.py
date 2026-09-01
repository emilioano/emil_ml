"""Scores an image via a trained CNN classifier's failed-class probability."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras

from emil_ml.config.registry import Component
from emil_ml.core.base import BasePredictor, PredictionResult
from emil_ml.utils import image_io
from emil_ml.utils.paths import for_component

_MODEL_CACHE: dict[tuple[str, float, int], keras.Model] = {}


def _load_cached(model_path: Path, image_size: int) -> keras.Model:
    """Cache loaded models by (path, mtime, image_size) so retraining invalidates the cache."""
    mtime = model_path.stat().st_mtime
    key = (str(model_path), mtime, image_size)
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = keras.models.load_model(model_path)
        _MODEL_CACHE[key] = model
    return model


class ClassifierPredictor(BasePredictor):
    """Loads the component's CNN classifier and scores images by failed-class probability.

    Takes the modality handler's decoded PIL Image and does its own
    square-resize + [0,1] normalization (specific to how this model expects
    input — not every method wants that, so it's not done upstream).
    """

    def __init__(self, component: Component) -> None:
        super().__init__(component)
        if not component.model_path:
            raise ValueError(f"Component {component.name!r} has no trained model recorded")
        paths = for_component(component.name)
        model_path = paths.resolve_model_path(component.model_path)
        self._model = _load_cached(model_path, component.image_size)

    def predict(self, prepared_input: Image.Image) -> PredictionResult:
        if self.component.anomaly_threshold is None:
            raise ValueError(f"Component {self.component.name!r} has no decision threshold set")

        x = image_io.normalize(prepared_input, self.component.image_size)
        batch = np.expand_dims(x, axis=0)
        failed_probability = float(self._model.predict(batch, verbose=0)[0, 0])
        approved_probability = 1.0 - failed_probability

        threshold = self.component.anomaly_threshold
        verdict = "failed" if failed_probability >= threshold else "approved"
        return PredictionResult(
            verdict=verdict,
            score=failed_probability,
            threshold=threshold,
            details={
                "failed_probability": failed_probability,
                "approved_probability": approved_probability,
            },
        )
