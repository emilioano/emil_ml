"""Phase 2b sanity check: onboard a 'classifier' component, train on synthetic
approved+failed images, inspect fresh samples, and print validation metrics
(especially recall on the failed class).

Uses small synthetic images so this runs quickly. Run with:
    python scripts/verify_classifier.py [mobilenet_v2|efficientnet_b0] [max|average]
"""

from __future__ import annotations

import io
import random
import sys

import numpy as np
from PIL import Image

from emil_ml.config.logging_config import configure_logging
from emil_ml.config.registry import ComponentRegistry
from emil_ml.pipeline.inspect import inspect
from emil_ml.training import onboard

IMAGE_SIZE = 64  # must be >= 32 for either base
COMPONENT_NAME = "test-classifier-widget"


def _png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def make_approved(rng: random.Random) -> bytes:
    # A consistent solid teal block with small per-image noise.
    base = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.int16)
    base[:, :] = [40, 160, 150]
    noise = np.array([[rng.gauss(0, 8) for _ in range(IMAGE_SIZE)] for _ in range(IMAGE_SIZE)])
    for c in range(3):
        base[:, :, c] = np.clip(base[:, :, c] + noise, 0, 255)
    return _png_bytes(base.astype(np.uint8))


def make_failed(rng: random.Random) -> bytes:
    # A visibly different pattern (bright random noise).
    arr = np.array(
        [[[rng.randint(0, 255) for _ in range(3)] for _ in range(IMAGE_SIZE)] for _ in range(IMAGE_SIZE)],
        dtype=np.uint8,
    )
    return _png_bytes(arr)


def main() -> None:
    configure_logging()
    from emil_ml.config.settings import DEFAULT_CLASSIFIER_BASE_MODEL, DEFAULT_CLASSIFIER_POOLING

    base_model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CLASSIFIER_BASE_MODEL
    pooling = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CLASSIFIER_POOLING
    rng = random.Random(42)
    registry = ComponentRegistry()

    if registry.get(COMPONENT_NAME) is not None:
        print(f"Deleting existing component {COMPONENT_NAME!r}...")
        registry.delete(COMPONENT_NAME)

    print(
        f"Creating component 'Test Classifier Widget' "
        f"(model_type='classifier', base_model={base_model!r}, pooling={pooling!r})..."
    )
    component = onboard.create_component(
        "Test Classifier Widget",
        image_size=IMAGE_SIZE,
        epochs=8,
        batch_size=8,
        model_type="classifier",
        base_model=base_model,
        pooling=pooling,
        registry=registry,
    )

    print("Generating synthetic training images (20 approved, 10 failed)...")
    approved = [(f"approved_{i}.png", make_approved(rng)) for i in range(20)]
    failed = [(f"failed_{i}.png", make_failed(rng)) for i in range(10)]
    onboard.add_training_images(component.name, approved=approved, failed=failed)

    print("Training (downloads ImageNet weights on first run)...")
    result = onboard.train_component(component.name, registry=registry)
    print(f"  -> decision threshold={result.threshold}")
    summary = {k: v for k, v in result.details.items() if k != "history"}
    print(f"  -> details: {summary}")

    trained = registry.get(component.name)
    assert trained is not None and trained.status == "ready"
    assert trained.anomaly_threshold == 0.5

    print("Inspecting a fresh approved-like image...")
    approved_result = inspect(make_approved(rng), component.name, registry=registry)
    print(f"  -> {approved_result}")

    print("Inspecting a fresh failed-like image...")
    failed_result = inspect(make_failed(rng), component.name, registry=registry)
    print(f"  -> {failed_result}")

    print("\nPhase 2b classifier pipeline OK.")


if __name__ == "__main__":
    main()
