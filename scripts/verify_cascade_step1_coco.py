"""Verifies Step 1 in isolation, now COCO-YOLO: the coarse detector as a
normal registry-driven component (model_type='coco_detector'), created/
trained/predicted through the exact same machinery as every other
model_type — no cascade-specific code involved yet. The direct successor
to scripts/verify_cascade_step1_resnet.py (kept, still valid — that
classifier is still a real, usable model_type, just no longer the
cascade's coarse stage; see core/detection/yolo_coco/__init__.py).

Run with: python scripts/verify_cascade_step1_coco.py
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
from emil_ml.core.cascade.categories import CATEGORY_ANIMAL, CATEGORY_HUMAN
from emil_ml.training import onboard

COMPONENT_DISPLAY_NAME = "Cascade Coarse COCO Test Component"


def main() -> None:
    configure_logging()
    registry = ComponentRegistry()
    all_pass = True
    component = onboard.create_component(
        COMPONENT_DISPLAY_NAME, model_type="coco_detector", registry=registry
    )
    name = component.name

    try:
        print("=== 1: create_component() accepts model_type='coco_detector' ===")
        ok1 = component.model_type == "coco_detector" and component.status == "created"
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

        predictor = registry_factory.get_predictor(component.modality, component.model_type, component)
        handler = registry_factory.get_modality_handler(component.modality, component)

        print("=== 3: detects a real human photo (skimage.data.astronaut) as 'person' -> category='human' ===")
        print("       (this is the exact case ImageNet-1k could never reliably produce — see resnet_coarse) ===")
        person_image = handler.load(Image.fromarray(data.astronaut()))
        person_result = predictor.predict(person_image)
        person_detections = person_result.details["detections"]
        person_hits = [d for d in person_detections if d["class"] == "person"]
        ok3 = (
            len(person_hits) >= 1
            and person_hits[0]["category"] == CATEGORY_HUMAN
            and person_hits[0]["confidence"] >= component.coco_confidence_threshold
            and person_hits[0]["box"] is not None
        )
        print(f"  detections: {[(d['class'], d['category'], round(d['confidence'], 4)) for d in person_detections]}")
        print(f"  verdict={person_result.verdict} (summary field only, not what the cascade acts on)")
        print(f"-> {'PASS' if ok3 else 'FAIL'}")
        all_pass &= ok3
        print()

        print("=== 4: detects a real cat photo (skimage.data.chelsea) as 'cat' -> category='animal' ===")
        cat_image = handler.load(Image.fromarray(data.chelsea()))
        cat_result = predictor.predict(cat_image)
        cat_detections = cat_result.details["detections"]
        cat_hits = [d for d in cat_detections if d["class"] == "cat"]
        ok4 = len(cat_hits) >= 1 and cat_hits[0]["category"] == CATEGORY_ANIMAL
        print(f"  detections: {[(d['class'], d['category'], round(d['confidence'], 4)) for d in cat_detections]}")
        print(f"-> {'PASS' if ok4 else 'FAIL'}")
        all_pass &= ok4
        print()

        print("=== 5: coco_confidence_threshold drops low-confidence candidates outright (no 'uncertain' box) ===")
        registry.update_settings(name, coco_confidence_threshold=0.999)  # nothing clears this bar
        component = registry.get(name)
        predictor_strict = registry_factory.get_predictor(component.modality, component.model_type, component)
        strict_result = predictor_strict.predict(person_image)
        ok5 = strict_result.details["detections"] == [] and strict_result.verdict == "uncertain"
        print(f"  threshold=0.999 detections={strict_result.details['detections']} verdict={strict_result.verdict}")
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
