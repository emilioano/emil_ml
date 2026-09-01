"""Coarse object DETECTOR: a frozen, off-the-shelf COCO-pretrained YOLO,
never fine-tuned — the cascade framework's Step 1 (see core/cascade),
replacing core/classification/resnet_coarse's ImageNet-1k classifier in
that role. Registered as model_type='coco_detector' in
core/registry_factory.py, reusing the exact same YOLO machinery
(ultralytics, core/detection/yolo/model.py's weight-fetching) the
component-specific defect detector (model_type='yolo') already uses —
just pointed at stock COCO weights instead of a fine-tuned checkpoint,
and never fine-tuned itself. "Training" it (trainer.py) is a fast no-op,
same reasoning as resnet_coarse's trainer.

WHY THIS REPLACED THE IMAGENET-1K COARSE CLASSIFIER FOR STEP 1: verified
directly (see resnet_coarse/imagenet_categories.py's own docstring) that
ImageNet-1k has no reliable "person" class — a real photo of a person
topped out around "bobsled" at ~26% confidence, not anything human-
related. That made the cascade's whole person -> face-recognition branch
permanently unreachable in practice, since Step 1 could never honestly
report the "human" category. COCO's 80 classes include "person" directly
(one of its most common, best-represented classes), plus everyday
animals (cat, dog, horse, bird, ...) and vehicles (car, truck, bus,
motorcycle, ...) — exactly the coarse categories this cascade cares
about — all with real bounding boxes, not just a whole-frame guess. YOLO
is also machinery this project already has deep, working investment in
(training, inference, weight-fetching), so this is a reuse, not a new
dependency.

Unlike the ImageNet classifier (single whole-frame label), this detector
finds POSSIBLY SEVERAL objects per frame, each with its own class, box,
and confidence — core/cascade/pipeline.py dispatches a specialist per
detected object (see its own module docstring), and the box is passed
through so a specialist (e.g. face recognition) can search within the
object's own region instead of the whole frame.

core/classification/resnet_coarse is NOT removed and remains a fully
valid, independently usable model_type — it's just no longer wired as
Step 1's default. It could still serve as an optional, secondary
FINE-GRAINED classifier layered after this detector for a category COCO
is too coarse about (e.g. COCO says "dog"; ImageNet can sometimes name
the breed) — that layering isn't built, this module docstring just notes
the door is open, per core/cascade/pipeline.py's own extensibility story.
"""
