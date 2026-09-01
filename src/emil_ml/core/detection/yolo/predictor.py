"""Detects objects via a trained YOLO model and applies the component's decision rule."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from ultralytics import YOLO

from emil_ml.config.registry import Component
from emil_ml.core.base import BasePredictor, PredictionResult
from emil_ml.utils.paths import for_component

# Low floor passed to Ultralytics itself — real thresholding happens below,
# against the component's (possibly session-overridden) threshold, not here.
# Keeps every candidate detection available so pipeline.inspect()'s generic
# threshold_override can re-derive the verdict without rerunning inference.
_INTERNAL_CONFIDENCE_FLOOR = 0.05

_MODEL_CACHE: dict[tuple[str, float], YOLO] = {}


def _load_cached(model_path: Path) -> YOLO:
    """Cache loaded models by (path, mtime) so retraining invalidates the cache."""
    mtime = model_path.stat().st_mtime
    key = (str(model_path), mtime)
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = YOLO(str(model_path))
        _MODEL_CACHE[key] = model
    return model


class YoloPredictor(BasePredictor):
    """Loads the component's fine-tuned YOLO model and scores images by detection.

    Takes the modality handler's decoded PIL Image directly — no forced
    square resize/normalization; Ultralytics does its own preprocessing
    (letterboxing etc.) and performs better on an undistorted image than the
    Keras-style predictors' square-resized input.
    """

    def __init__(self, component: Component) -> None:
        super().__init__(component)
        if not component.model_path:
            raise ValueError(f"Component {component.name!r} has no trained model recorded")
        paths = for_component(component.name)
        model_path = paths.resolve_model_path(component.model_path)
        self._model = _load_cached(model_path)

    def predict(self, prepared_input: Image.Image) -> PredictionResult:
        if self.component.anomaly_threshold is None:
            raise ValueError(f"Component {self.component.name!r} has no confidence threshold set")
        threshold = self.component.anomaly_threshold

        results = self._model.predict(prepared_input, conf=_INTERNAL_CONFIDENCE_FLOOR, verbose=False)[0]

        detections = []
        for box in results.boxes:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = results.names.get(class_id, str(class_id))
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            detections.append(
                {"class": class_name, "confidence": confidence, "box": [x1, y1, x2, y2]}
            )

        # Computed from ALL candidate detections (not pre-filtered by
        # `threshold`), so a session-only threshold override in the UI can
        # correctly re-derive the verdict/score without seeing stale data —
        # `details["detections"]` also isn't filtered, for the same reason;
        # the UI decides what to draw by comparing each detection's own
        # confidence against whatever threshold is currently active.
        max_confidence = max((d["confidence"] for d in detections), default=0.0)
        found = max_confidence >= threshold

        if self.component.decision_rule == "presence":
            # Looking for something that shouldn't be there (a defect, a
            # foreign object) — finding it means "failed".
            verdict = "failed" if found else "approved"
            score = max_confidence  # higher = more clearly failed
        else:
            # "absence": looking for something that should be there (e.g. a
            # required part) — not finding it means "failed". Score still
            # follows "higher = more failed" so the generic threshold
            # override (which assumes that direction) works correctly; it
            # isn't literally "confidence" here, it's an inverted failure
            # signal (found confidently -> low failure score).
            verdict = "approved" if found else "failed"
            score = (1.0 - max_confidence) if found else 1.0

        return PredictionResult(
            verdict=verdict, score=score, threshold=threshold, details={"detections": detections}
        )
