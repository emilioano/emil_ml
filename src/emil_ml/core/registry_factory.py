"""Maps a component's (modality, model_type) to its concrete implementation.

This is the *only* place in the system that knows which (modality,
model_type) maps to which classes. `pipeline.inspect`, `training.onboard`,
and the Streamlit UI all go through `get_modality_handler` / `get_trainer` /
`get_predictor` and never branch on modality or model_type themselves.
Adding a new method or modality: implement the relevant base class and add
one factory function + one line to the maps below.

Every method's own heavy framework import (TensorFlow for autoencoder/
classifier, PyTorch/Ultralytics for YOLO, anomalib for PatchCore) is deferred
to inside these factory functions, not imported at this module's top level.
This isn't just about PatchCore's optional dependency (see its own trainer/
predictor modules for that story) — TensorFlow and PyTorch's Triton JIT
compiler have been observed to crash when loaded into the same process on at
least one real GPU/driver combination (a segfault inside libtriton.so,
happening only when TensorFlow had also been imported). Since this module
used to import every trainer/predictor eagerly, simply importing
`registry_factory` — which every page does — loaded every framework into the
process regardless of which model_type was actually used. Deferring these
imports means a session that only ever trains/inspects YOLO components never
loads TensorFlow at all (and vice versa), which is the actual fix for that
crash, not just a lucky side effect.
"""

from __future__ import annotations

from typing import Callable

from emil_ml.config.registry import Component
from emil_ml.core.base import BasePredictor, BaseTrainer
from emil_ml.core.modality.base import BaseModalityHandler
from emil_ml.core.modality.image_handler import ImageModalityHandler

_MODALITY_HANDLERS: dict[str, type[BaseModalityHandler]] = {
    "image": ImageModalityHandler,
}


def _autoencoder_trainer() -> BaseTrainer:
    from emil_ml.core.anomaly.autoencoder.trainer import AutoencoderTrainer

    return AutoencoderTrainer()


def _autoencoder_predictor(component: Component) -> BasePredictor:
    from emil_ml.core.anomaly.autoencoder.predictor import AutoencoderPredictor

    return AutoencoderPredictor(component)


def _classifier_trainer() -> BaseTrainer:
    from emil_ml.core.classification.cnn_classifier.trainer import ClassifierTrainer

    return ClassifierTrainer()


def _classifier_predictor(component: Component) -> BasePredictor:
    from emil_ml.core.classification.cnn_classifier.predictor import ClassifierPredictor

    return ClassifierPredictor(component)


def _yolo_trainer() -> BaseTrainer:
    from emil_ml.core.detection.yolo.trainer import YoloTrainer

    return YoloTrainer()


def _yolo_predictor(component: Component) -> BasePredictor:
    from emil_ml.core.detection.yolo.predictor import YoloPredictor

    return YoloPredictor(component)


def _patchcore_trainer() -> BaseTrainer:
    from emil_ml.core.anomaly.patchcore.trainer import PatchCoreTrainer

    return PatchCoreTrainer()


def _patchcore_predictor(component: Component) -> BasePredictor:
    from emil_ml.core.anomaly.patchcore.predictor import PatchCorePredictor

    return PatchCorePredictor(component)


def _isolation_forest_trainer() -> BaseTrainer:
    from emil_ml.core.anomaly.isolation_forest.trainer import IsolationForestTrainer

    return IsolationForestTrainer()


def _isolation_forest_predictor(component: Component) -> BasePredictor:
    from emil_ml.core.anomaly.isolation_forest.predictor import IsolationForestPredictor

    return IsolationForestPredictor(component)


def _resnet_classifier_trainer() -> BaseTrainer:
    from emil_ml.core.classification.resnet_coarse.trainer import ResNetCoarseTrainer

    return ResNetCoarseTrainer()


def _resnet_classifier_predictor(component: Component) -> BasePredictor:
    from emil_ml.core.classification.resnet_coarse.predictor import ResNetCoarsePredictor

    return ResNetCoarsePredictor(component)


def _coco_detector_trainer() -> BaseTrainer:
    from emil_ml.core.detection.yolo_coco.trainer import CocoCoarseTrainer

    return CocoCoarseTrainer()


def _coco_detector_predictor(component: Component) -> BasePredictor:
    from emil_ml.core.detection.yolo_coco.predictor import CocoCoarsePredictor

    return CocoCoarsePredictor(component)


_TRAINER_FACTORIES: dict[tuple[str, str], Callable[[], BaseTrainer]] = {
    ("image", "autoencoder"): _autoencoder_trainer,
    ("image", "classifier"): _classifier_trainer,
    ("image", "yolo"): _yolo_trainer,
    ("image", "patchcore"): _patchcore_trainer,
    ("image", "isolation_forest"): _isolation_forest_trainer,
    ("image", "resnet_classifier"): _resnet_classifier_trainer,
    ("image", "coco_detector"): _coco_detector_trainer,
}

_PREDICTOR_FACTORIES: dict[tuple[str, str], Callable[[Component], BasePredictor]] = {
    ("image", "autoencoder"): _autoencoder_predictor,
    ("image", "classifier"): _classifier_predictor,
    ("image", "yolo"): _yolo_predictor,
    ("image", "patchcore"): _patchcore_predictor,
    ("image", "isolation_forest"): _isolation_forest_predictor,
    ("image", "resnet_classifier"): _resnet_classifier_predictor,
    ("image", "coco_detector"): _coco_detector_predictor,
}

# Model types whose trainer requires at least one failed-labeled example (the
# autoencoder is fine with approved-only; the classifier is supervised and
# needs both classes). Queried by the onboarding UI to validate uploads
# before submitting, instead of branching on model_type there directly.
_REQUIRES_FAILED_EXAMPLES = {"classifier"}

# Valid-but-not-implemented-yet values, used only to give clearer error messages.
_NOT_YET_IMPLEMENTED_MODALITIES = {"text"}
_NOT_YET_IMPLEMENTED_MODEL_TYPES: set[str] = set()


def get_modality_handler(modality: str, component: Component) -> BaseModalityHandler:
    """Return a modality handler instance bound to `component`."""
    try:
        handler_cls = _MODALITY_HANDLERS[modality]
    except KeyError:
        raise _unavailable_modality_error(modality) from None
    return handler_cls(component)


def get_trainer(modality: str, model_type: str) -> BaseTrainer:
    """Return a trainer instance for the (modality, model_type) combination."""
    key = (modality, model_type)
    try:
        factory = _TRAINER_FACTORIES[key]
    except KeyError:
        raise _unavailable_combo_error(modality, model_type) from None
    return factory()


def get_predictor(modality: str, model_type: str, component: Component) -> BasePredictor:
    """Return a predictor instance bound to `component` for the (modality, model_type) combination."""
    key = (modality, model_type)
    try:
        factory = _PREDICTOR_FACTORIES[key]
    except KeyError:
        raise _unavailable_combo_error(modality, model_type) from None
    return factory(component)


def requires_failed_examples(model_type: str) -> bool:
    """Whether this model_type's trainer needs at least one failed-labeled example."""
    return model_type in _REQUIRES_FAILED_EXAMPLES


def _unavailable_modality_error(modality: str) -> ValueError:
    if modality in _NOT_YET_IMPLEMENTED_MODALITIES:
        return ValueError(f"modality {modality!r} is not implemented yet")
    return ValueError(f"Unknown modality: {modality!r}")


def _unavailable_combo_error(modality: str, model_type: str) -> ValueError:
    if modality in _NOT_YET_IMPLEMENTED_MODALITIES:
        return ValueError(f"modality {modality!r} is not implemented yet")
    if model_type in _NOT_YET_IMPLEMENTED_MODEL_TYPES:
        return ValueError(f"model_type {model_type!r} is not implemented yet")
    return ValueError(f"No implementation registered for (modality={modality!r}, model_type={model_type!r})")
