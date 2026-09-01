"""Scores a frame with a frozen ImageNet-1k ResNet-50 and maps the top-1
label to a coarse category — see imagenet_categories.py for the mapping
and its documented limitations.

Reuses PredictionResult (core/base.py) exactly as every other predictor
does, but repurposes `verdict` to carry the coarse category string (e.g.
"animal", "vehicle", "uncertain") instead of "approved"/"failed" — there
is no pass/fail concept for a coarse classifier. This is a deliberate,
narrow reuse of the existing uniform result shape (per the cascade
framework's design: one common result dataclass, method-specific meaning
in `details`) rather than inventing a parallel type; nothing downstream
of registry_factory.get_predictor() assumes `verdict` is literally
"approved"/"failed" — pipeline/inspect.py's threshold_override rewrite is
the only place that does, and it's never invoked for this model_type (the
cascade's own pipeline.py calls get_predictor() directly, bypassing
pipeline/inspect.py entirely — see cascade/pipeline.py).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from emil_ml.config.registry import Component
from emil_ml.core.base import BasePredictor, PredictionResult
from emil_ml.core.classification.resnet_coarse.imagenet_categories import (
    CATEGORY_UNCERTAIN,
    categorize,
)

_TOP_K = 5
_MODEL: Any = None  # lazily loaded, shared across every resnet_classifier component — fixed pretrained weights, nothing per-component to isolate


def _load_model() -> Any:
    global _MODEL
    if _MODEL is None:
        from tensorflow.keras.applications.resnet50 import ResNet50

        _MODEL = ResNet50(weights="imagenet")
    return _MODEL


class ResNetCoarsePredictor(BasePredictor):
    """Loads the shared, frozen ImageNet-1k ResNet-50 and classifies a frame."""

    def __init__(self, component: Component) -> None:
        super().__init__(component)
        self._model = _load_model()

    def predict(self, prepared_input: Image.Image) -> PredictionResult:
        from tensorflow.keras.applications.resnet50 import decode_predictions, preprocess_input

        resized = prepared_input.resize((224, 224))
        batch = np.expand_dims(np.array(resized).astype("float32"), axis=0)
        batch = preprocess_input(batch)
        raw_preds = self._model.predict(batch, verbose=0)
        top_k = decode_predictions(raw_preds, top=_TOP_K)[0]  # [(wnid, label, prob), ...]

        top_label, top_confidence = top_k[0][1], float(top_k[0][2])
        threshold = self.component.resnet_confidence_threshold
        category = categorize(top_label) if top_confidence >= threshold else CATEGORY_UNCERTAIN

        return PredictionResult(
            verdict=category,
            score=top_confidence,
            threshold=threshold,
            details={
                "imagenet_label": top_label,
                "confidence": top_confidence,
                "top_k": [{"label": label, "confidence": float(prob)} for _, label, prob in top_k],
            },
        )
