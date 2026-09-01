"""Verifies Step 1 in isolation: the ResNet-50 coarse classifier as a
normal registry-driven component (model_type='resnet_classifier'),
created/trained/predicted through the exact same machinery as every other
model_type — no cascade-specific code involved yet.

Run with: python scripts/verify_cascade_step1_resnet.py
"""

from __future__ import annotations

import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from PIL import Image
from skimage import data

from emil_ml.config.logging_config import configure_logging
from emil_ml.config.registry import ComponentRegistry
from emil_ml.core import registry_factory
from emil_ml.training import onboard

COMPONENT_DISPLAY_NAME = "Cascade Coarse Test Component"


def main() -> None:
    configure_logging()
    registry = ComponentRegistry()
    all_pass = True
    component = onboard.create_component(
        COMPONENT_DISPLAY_NAME, model_type="resnet_classifier", registry=registry
    )
    name = component.name

    try:
        print("=== 1: create_component() accepts model_type='resnet_classifier' ===")
        ok1 = component.model_type == "resnet_classifier" and component.status == "created"
        print(f"  status={component.status} model_type={component.model_type}")
        print(f"-> {'PASS' if ok1 else 'FAIL'}")
        all_pass &= ok1
        print()

        print("=== 2: train_component() is a fast no-op that reaches status='ready' ===")
        result = onboard.train_component(name, registry=registry)
        component = registry.get(name)
        ok2 = component.status == "ready" and result.model_path is None and result.threshold is None
        print(f"  status={component.status} model_path={result.model_path} threshold={result.threshold}")
        print(f"-> {'PASS' if ok2 else 'FAIL'}")
        all_pass &= ok2
        print()

        print("=== 3: predicts a real cat photo (skimage.data.chelsea) as category='animal' ===")
        predictor = registry_factory.get_predictor(component.modality, component.model_type, component)
        handler = registry_factory.get_modality_handler(component.modality, component)
        cat_image = handler.load(Image.fromarray(data.chelsea()))
        cat_result = predictor.predict(cat_image)
        ok3 = cat_result.verdict == "animal" and cat_result.details["imagenet_label"] in (
            "Egyptian_cat", "tiger_cat", "tabby", "lynx", "Persian_cat",
        )
        print(f"  verdict={cat_result.verdict} label={cat_result.details['imagenet_label']} "
              f"confidence={cat_result.details['confidence']:.4f}")
        print(f"-> {'PASS' if ok3 else 'FAIL'}")
        all_pass &= ok3
        print()

        print("=== 4: a real human photo (skimage.data.astronaut) demonstrates the documented ===")
        print("       ImageNet-1k 'no generic person class' limitation honestly (not silently 'human') ===")
        person_image = handler.load(Image.fromarray(data.astronaut()))
        person_result = predictor.predict(person_image)
        # We do NOT assert verdict == "human" here — that's exactly the limitation
        # documented in imagenet_categories.py. We assert the coarse stage still
        # behaves sanely: a real label, a real confidence, never a crash/blank.
        ok4 = (
            person_result.verdict in ("animal", "vehicle", "other", "uncertain")
            and person_result.details["imagenet_label"]
            and 0.0 <= person_result.details["confidence"] <= 1.0
        )
        print(f"  verdict={person_result.verdict} label={person_result.details['imagenet_label']} "
              f"confidence={person_result.details['confidence']:.4f}")
        print(f"  (this is the documented limitation, not a bug — see imagenet_categories.py)")
        print(f"-> {'PASS' if ok4 else 'FAIL'}")
        all_pass &= ok4
        print()

        print("=== 5: resnet_confidence_threshold gates low-confidence predictions to 'uncertain' ===")
        registry.update_settings(name, resnet_confidence_threshold=0.999)  # nothing clears this bar
        component = registry.get(name)
        predictor_strict = registry_factory.get_predictor(component.modality, component.model_type, component)
        strict_result = predictor_strict.predict(cat_image)
        ok5 = strict_result.verdict == "uncertain"
        print(f"  threshold=0.999 verdict={strict_result.verdict}")
        print(f"-> {'PASS' if ok5 else 'FAIL'}")
        all_pass &= ok5
        print()

        print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED — see above'}")
    finally:
        if registry.get(name) is not None:
            registry.soft_delete(name)
            from emil_ml.core import component_deletion

            component_deletion.permanently_delete_component(name, registry=registry)


if __name__ == "__main__":
    main()
