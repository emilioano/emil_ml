"""Detects objects with a frozen, stock COCO-pretrained YOLO and maps
each detection's COCO class to a coarse category — see coco_categories.py
for the mapping.

Reuses PredictionResult (core/base.py) the same deliberate, narrow way
resnet_coarse's predictor did: `verdict` carries the HIGHEST-confidence
detection's category as a human-readable summary only (or
CATEGORY_UNCERTAIN if nothing cleared the confidence floor) — it is NOT
what core/cascade/pipeline.py actually acts on. Unlike the ImageNet
classifier this replaced, this predictor can find MULTIPLE objects in one
frame, so the authoritative, structured result is the full list in
`details["detections"]`: one dict per detection, each already carrying
its own mapped `"category"` (this predictor owns that mapping, the same
split resnet_coarse used — a coarse method maps its own raw vocabulary to
the shared category vocabulary itself; core/cascade never needs to know
COCO exists). core/cascade/pipeline.py is what actually iterates this
list and dispatches a specialist per object — see its own module
docstring for why iterating every detection (not just the top one) is
the deliberate choice here.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from emil_ml.config.registry import Component
from emil_ml.core.base import BasePredictor, PredictionResult
from emil_ml.core.cascade.categories import CATEGORY_UNCERTAIN
from emil_ml.core.detection.yolo.model import load_yolo_model
from emil_ml.core.detection.yolo_coco.coco_categories import categorize

# Low floor passed to Ultralytics itself, same reasoning as
# core/detection/yolo/predictor.py's own _INTERNAL_CONFIDENCE_FLOOR — real
# thresholding happens below, against the component's own
# coco_confidence_threshold, not here.
_INTERNAL_CONFIDENCE_FLOOR = 0.05

_MODEL_CACHE: dict[str, Any] = {}  # keyed by yolo_model_variant — fixed pretrained weights, shared across every coco_detector component


def _load_model(model_variant: str) -> Any:
    model = _MODEL_CACHE.get(model_variant)
    if model is None:
        model = load_yolo_model(model_variant)
        _MODEL_CACHE[model_variant] = model
    return model


class CocoCoarsePredictor(BasePredictor):
    """Loads the shared, frozen COCO-pretrained YOLO and detects objects in a frame."""

    def __init__(self, component: Component) -> None:
        super().__init__(component)
        self._model = _load_model(component.yolo_model_variant)

    def predict(self, prepared_input: Image.Image) -> PredictionResult:
        threshold = self.component.coco_confidence_threshold
        results = self._model.predict(prepared_input, conf=_INTERNAL_CONFIDENCE_FLOOR, verbose=False)[0]

        detections: list[dict[str, Any]] = []
        for box in results.boxes:
            confidence = float(box.conf[0])
            if confidence < threshold:
                continue
            class_id = int(box.cls[0])
            class_name = results.names.get(class_id, str(class_id))
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            detections.append(
                {
                    "class": class_name,
                    "category": categorize(class_name),
                    "confidence": confidence,
                    "box": [x1, y1, x2, y2],
                }
            )
        detections.sort(key=lambda d: d["confidence"], reverse=True)

        top = detections[0] if detections else None
        return PredictionResult(
            verdict=top["category"] if top else CATEGORY_UNCERTAIN,
            score=top["confidence"] if top else 0.0,
            threshold=threshold,
            details={"detections": detections},
        )
