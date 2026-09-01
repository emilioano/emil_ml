"""Extracts CNN embeddings from a component's training images.

Read-only: this loads the same training/approved and training/failed images
the autoencoder and classifier train on, and runs them through a frozen
ImageNet-pretrained MobileNetV2 (the same backbone family the classifier
uses, via a global-average-pooled feature vector instead of a
classification head) to get one feature vector per image. Nothing here
trains, fits, or writes any model artifact — see projection.py for what
happens to these vectors next.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tensorflow import keras

from emil_ml.utils import image_io
from emil_ml.utils.paths import ComponentPaths

# MobileNetV2's native ImageNet input size. Independent of whatever
# image_size a component's trained model uses — this is a fixed-backbone
# feature extractor, not that model.
DEFAULT_IMAGE_SIZE = 224

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

_embedding_model_cache: dict[int, keras.Model] = {}


def _get_embedding_model(image_size: int) -> keras.Model:
    """Frozen MobileNetV2 + global average pooling, cached per image_size."""
    if image_size not in _embedding_model_cache:
        base = keras.applications.MobileNetV2(
            input_shape=(image_size, image_size, 3),
            include_top=False,
            weights="imagenet",
            pooling="avg",
        )
        base.trainable = False
        _embedding_model_cache[image_size] = base
    return _embedding_model_cache[image_size]


def _list_image_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS)


@dataclass(frozen=True)
class LabeledEmbeddings:
    """Embeddings + their approved/failed labels, one row per image."""

    embeddings: np.ndarray  # (N, feature_dim) float32
    labels: list[str]  # "approved" | "failed", len N, same order as embeddings' rows
    filenames: list[str]  # len N, same order — for hover text / debugging in the UI


def extract_embeddings(paths: ComponentPaths, *, image_size: int = DEFAULT_IMAGE_SIZE) -> LabeledEmbeddings:
    """Embed a component's approved (+ failed, if any) training images.

    Returns an empty `LabeledEmbeddings` if the component has no training
    images yet, rather than raising — callers (the diagnostics page) should
    check `len(result.labels) == 0` and show a clear message instead of a
    stack trace.
    """
    model = _get_embedding_model(image_size)

    embeddings: list[np.ndarray] = []
    labels: list[str] = []
    filenames: list[str] = []

    for label, directory in (("approved", paths.training_approved_dir), ("failed", paths.training_failed_dir)):
        if not directory.exists():
            continue
        image_paths = _list_image_files(directory)
        if not image_paths:
            continue
        # MobileNetV2 expects roughly [-1, 1] input; image_io.preprocess()
        # gives [0, 1], so rescale rather than reusing its normalization.
        batch = np.stack([image_io.preprocess(p, image_size) for p in image_paths]) * 2.0 - 1.0
        batch_embeddings = model.predict(batch, verbose=0)
        embeddings.append(batch_embeddings)
        labels.extend([label] * len(image_paths))
        filenames.extend(p.name for p in image_paths)

    if not embeddings:
        return LabeledEmbeddings(embeddings=np.empty((0, 0), dtype=np.float32), labels=[], filenames=[])

    return LabeledEmbeddings(embeddings=np.concatenate(embeddings, axis=0), labels=labels, filenames=filenames)


def embed_image(image: image_io.ImageInput, *, image_size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """Embed a single image — the single-image counterpart to extract_embeddings().

    For anything that needs one image's embedding at inference time (e.g.
    core/anomaly/isolation_forest/predictor.py) via the exact same backbone
    and preprocessing extract_embeddings() uses for training, so the two are
    guaranteed consistent rather than two independent implementations that
    could quietly drift apart.
    """
    model = _get_embedding_model(image_size)
    batch = np.expand_dims(image_io.preprocess(image, image_size) * 2.0 - 1.0, axis=0)
    return model.predict(batch, verbose=0)[0]
