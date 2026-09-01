"""Builds a transfer-learning CNN for binary approved/failed classification.

Small labeled datasets (tens of images per class) overfit almost immediately
if trained from scratch, so this freezes a pretrained ImageNet base and only
trains a small classification head on top. Output is a single sigmoid unit:
the probability of "failed" (label 1).
"""

from __future__ import annotations

from tensorflow import keras
from keras import layers

DEFAULT_BASE_MODEL = "mobilenet_v2"
DEFAULT_POOLING = "average"

_BASE_MODEL_BUILDERS = {
    "mobilenet_v2": keras.applications.MobileNetV2,
    "efficientnet_b0": keras.applications.EfficientNetB0,
}

_POOLING_LAYERS = {
    "max": layers.GlobalMaxPooling2D,
    "average": layers.GlobalAveragePooling2D,
}

# (scale, offset) applied to our [0, 1]-normalized input so it matches what
# each pretrained base expects as raw input — expressed as a plain affine
# transform so it can be a built-in Rescaling layer (cleanly serializable)
# rather than a Lambda wrapping that base's own preprocess_input function.
_BASE_INPUT_RESCALE = {
    "mobilenet_v2": (2.0, -1.0),  # -> [-1, 1], matches mobilenet_v2.preprocess_input
    "efficientnet_b0": (255.0, 0.0),  # -> [0, 255]; EfficientNet normalizes internally
}


def build_classifier(
    image_size: int,
    *,
    base_model_name: str = DEFAULT_BASE_MODEL,
    pooling: str = DEFAULT_POOLING,
    dense_units: int = 128,
    dropout: float = 0.3,
    augment: bool = True,
    augmentation_strength: float = 0.06,
) -> keras.Model:
    """Frozen pretrained base + global pool + dense + sigmoid head for binary classification.

    `pooling`: "max" keeps whatever the strongest local activation was,
    regardless of how little of the image it covers — good when a defect is
    small/localized, since average pooling would dilute it across the whole
    spatial extent. "average" smooths over the whole feature map instead —
    can generalize better when the discriminative signal isn't localized.

    `augment` adds random flip/rotation/zoom/brightness layers before the
    base model; like Dropout, these are Keras layers that are automatically
    no-ops when the model is called with training=False (i.e. during
    predict()/evaluate()), so no separate inference-time handling is needed.
    `augmentation_strength` (0-1) scales the rotation/zoom/brightness factors
    — kept mild by default since aggressive augmentation can distort or crop
    away a subtle, localized defect more often than it helps the model
    generalize. Horizontal flip isn't scaled by this — it can't destroy that
    kind of signal, only mirror it.
    """
    if base_model_name not in _BASE_MODEL_BUILDERS:
        raise ValueError(
            f"Unknown base_model_name {base_model_name!r}; must be one of {list(_BASE_MODEL_BUILDERS)}"
        )
    if pooling not in _POOLING_LAYERS:
        raise ValueError(f"Unknown pooling {pooling!r}; must be one of {list(_POOLING_LAYERS)}")

    base_builder = _BASE_MODEL_BUILDERS[base_model_name]
    scale, offset = _BASE_INPUT_RESCALE[base_model_name]

    base = base_builder(input_shape=(image_size, image_size, 3), include_top=False, weights="imagenet")
    base.trainable = False

    inputs = keras.Input(shape=(image_size, image_size, 3), name="image")
    x = inputs
    if augment:
        x = layers.RandomFlip("horizontal")(x)
        if augmentation_strength > 0:
            x = layers.RandomRotation(augmentation_strength)(x)
            x = layers.RandomZoom(augmentation_strength)(x)
            x = layers.RandomBrightness(augmentation_strength)(x)
    x = layers.Rescaling(scale=scale, offset=offset)(x)
    x = base(x, training=False)
    x = _POOLING_LAYERS[pooling]()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="failed_probability")(x)

    model = keras.Model(inputs, outputs, name="emil_cnn_classifier")
    # Exposed so the trainer can run a second fine-tuning phase (unfreeze the
    # base's top layers, continue training at a low LR) after head training —
    # frozen ImageNet features alone often can't separate subtle,
    # domain-specific defects. Only used during training; a model reloaded
    # via keras.models.load_model() for inference doesn't need it.
    model.base = base
    return model
