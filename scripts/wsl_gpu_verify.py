"""Verifies TensorFlow and PyTorch both see the GPU in the same venv.

Run after scripts/wsl_gpu_setup.sh. Deliberately exercises a Conv2DTranspose
forward+backward pass, not just a Conv2D or a device-detection check — that's
the op class that actually exposed the missing-libdevice gap (see
wsl_gpu_setup.sh's header comment for the full story).
"""

import emil_ml  # noqa: F401 (sets LD_LIBRARY_PATH to cover every nvidia/*/lib dir)
import tensorflow as tf

print("TensorFlow GPUs:", tf.config.list_physical_devices("GPU"))

with tf.device("/GPU:0"):
    x = tf.random.normal((8, 16, 16, 32))
    layer = tf.keras.layers.Conv2DTranspose(16, 3, strides=2, padding="same")
    with tf.GradientTape() as tape:
        y = layer(x)
        loss = tf.reduce_mean(y**2)
    tape.gradient(loss, layer.trainable_variables)
    print("Conv2DTranspose fwd+bwd OK, output shape:", y.shape)

import torch  # noqa: E402

print("PyTorch CUDA available:", torch.cuda.is_available())
