"""Scores an image by autoencoder reconstruction error against a threshold."""

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


class AutoencoderPredictor(BasePredictor):
    """Loads the component's autoencoder and scores images by reconstruction error.

    Takes the modality handler's decoded PIL Image and does its own
    square-resize + [0,1] normalization — that's specific to how this model
    was trained, not something every method wants (YOLO in particular does
    its own preprocessing and would rather not have the image force-resized
    to a square first).
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
            raise ValueError(f"Component {self.component.name!r} has no anomaly threshold set")

        x = image_io.normalize(prepared_input, self.component.image_size)
        batch = np.expand_dims(x, axis=0)
        reconstruction = self._model.predict(batch, verbose=0)[0]
        squared_error = np.square(x - reconstruction)  # (H, W, C)

        # Must match the score_method used at training time (that's what the
        # stored threshold was calibrated against) — see trainer.py.
        if self.component.score_method == "local_max":
            score = float(np.max(np.mean(squared_error, axis=2)))
        else:
            score = float(np.mean(squared_error))

        threshold = self.component.anomaly_threshold
        verdict = "approved" if score <= threshold else "failed"
        return PredictionResult(verdict=verdict, score=score, threshold=threshold, details={})
