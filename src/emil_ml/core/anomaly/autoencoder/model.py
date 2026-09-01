"""Builds the convolutional autoencoder architecture used for anomaly detection."""

from __future__ import annotations

from tensorflow import keras
from keras import layers

_ENCODER_FILTERS = (32, 64, 128, 256)
_DECODER_FILTERS = (128, 64, 32, 16)


def build_autoencoder(image_size: int, latent_dim: int) -> keras.Model:
    """Build a symmetric conv autoencoder: image_size -> latent_dim -> image_size.

    Downsamples with strided convolutions (one block per entry in
    `_ENCODER_FILTERS`), so `image_size` must be a multiple of 16.
    """
    if image_size % 16 != 0:
        raise ValueError(f"image_size must be a multiple of 16, got {image_size}")

    inputs = keras.Input(shape=(image_size, image_size, 3), name="image")

    x = inputs
    for filters in _ENCODER_FILTERS:
        x = layers.Conv2D(filters, 3, strides=2, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)

    bottleneck_shape = tuple(x.shape[1:])  # (H/16, W/16, channels)
    x = layers.Flatten()(x)
    latent = layers.Dense(latent_dim, activation="relu", name="latent")(x)

    x = layers.Dense(
        bottleneck_shape[0] * bottleneck_shape[1] * bottleneck_shape[2], activation="relu"
    )(latent)
    x = layers.Reshape(bottleneck_shape)(x)

    for filters in _DECODER_FILTERS:
        x = layers.Conv2DTranspose(filters, 3, strides=2, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)

    outputs = layers.Conv2D(3, 3, padding="same", activation="sigmoid", name="reconstruction")(x)

    return keras.Model(inputs, outputs, name="emil_autoencoder")
