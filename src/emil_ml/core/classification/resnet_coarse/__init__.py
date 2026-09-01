"""Coarse image classifier: a frozen, off-the-shelf ImageNet-1k ResNet-50,
never fine-tuned. Registered as model_type='resnet_classifier' in
core/registry_factory.py, so it's created/managed exactly like every other
component (same onboarding flow, same BasePredictor interface); "training"
it (trainer.py) is a fast no-op, since there are no weights to fit.

NO LONGER THE CASCADE'S STEP 1 — see core/detection/yolo_coco/__init__.py
for the replacement and why: this classifier gives one whole-frame label
with no reliable "human" category (ImageNet-1k has no generic person
class — see imagenet_categories.py's own docstring), which permanently
starved the cascade's person -> face-recognition branch. COCO-YOLO
(model_type='coco_detector') replaced it as the cascade's coarse stage,
with real bounding boxes and a person class besides.

This module is NOT removed and remains fully valid/usable on its own —
still a real, working classifier, still registered in registry_factory.
It could still serve later as an optional, secondary FINE-GRAINED
classifier layered after COCO-YOLO for a category COCO is too coarse
about (COCO says "dog"; this can sometimes name the breed) — not built,
just a documented option; nothing currently wires it into the cascade.

See imagenet_categories.py's own docstring for the "human" limitation in
full, and settings.py's DEFAULT_RESNET_CONFIDENCE_THRESHOLD comment for
the low-confidence "uncertain" fallback.
"""
