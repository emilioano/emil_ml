"""Phase 2 sanity check: onboard a component, train on synthetic images, inspect.

Uses small synthetic images (no TF/GPU-heavy settings) so this runs in
seconds. Run with: python scripts/verify_pipeline.py
"""

from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image

from emil_ml.config.logging_config import configure_logging
from emil_ml.config.registry import ComponentRegistry
from emil_ml.pipeline.inspect import inspect
from emil_ml.training import onboard

IMAGE_SIZE = 32
COMPONENT_NAME = "test-widget"


def _png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def make_approved(rng: random.Random) -> bytes:
    # A consistent base pattern (teal square) with small per-image noise.
    base = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.int16)
    base[:, :] = [40, 160, 150]
    noise = np.array(
        [[rng.gauss(0, 8) for _ in range(IMAGE_SIZE)] for _ in range(IMAGE_SIZE)]
    )
    for c in range(3):
        base[:, :, c] = np.clip(base[:, :, c] + noise, 0, 255)
    return _png_bytes(base.astype(np.uint8))


def make_failed(rng: random.Random) -> bytes:
    # A visibly different pattern (bright random noise) the model never trained on.
    arr = np.array(
        [[[rng.randint(0, 255) for _ in range(3)] for _ in range(IMAGE_SIZE)] for _ in range(IMAGE_SIZE)],
        dtype=np.uint8,
    )
    return _png_bytes(arr)


def main() -> None:
    configure_logging()
    rng = random.Random(42)
    registry = ComponentRegistry()

    if registry.get(COMPONENT_NAME) is not None:
        print(f"Deleting existing component {COMPONENT_NAME!r}...")
        registry.delete(COMPONENT_NAME)

    print("Creating component 'Test Widget'...")
    component = onboard.create_component(
        "Test Widget",
        image_size=IMAGE_SIZE,
        epochs=5,
        latent_dim=8,
        batch_size=4,
        registry=registry,
    )

    print("Generating synthetic training images...")
    approved = [(f"approved_{i}.png", make_approved(rng)) for i in range(20)]
    failed = [(f"failed_{i}.png", make_failed(rng)) for i in range(5)]
    onboard.add_training_images(component.name, approved=approved, failed=failed)

    print("Training...")
    result = onboard.train_component(component.name, registry=registry)
    print(f"  -> threshold={result.threshold:.6f}")
    summary = {k: v for k, v in result.details.items() if k not in ("approved_errors", "failed_errors", "history")}
    print(f"  -> details summary: {summary}")

    trained = registry.get(component.name)
    assert trained is not None and trained.status == "ready"
    assert trained.anomaly_threshold is not None

    print("Inspecting a fresh approved-like image...")
    approved_result = inspect(make_approved(rng), component.name, registry=registry)
    print(f"  -> {approved_result}")

    print("Inspecting a fresh failed-like image...")
    failed_result = inspect(make_failed(rng), component.name, registry=registry)
    print(f"  -> {failed_result}")

    print("\nPhase 2 pipeline OK.")


if __name__ == "__main__":
    main()
