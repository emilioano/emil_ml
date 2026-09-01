"""COCO class name -> coarse category mapping — the data half of Step 1's
new COCO-YOLO detector, replacing resnet_coarse/imagenet_categories.py in
that role (see this package's __init__.py for why).

An explicit dict, not the substring/keyword matching
imagenet_categories.py needed: COCO has only 80 classes, each already a
short, clean, unambiguous name (no ~120-breed families to collapse via
keywords, and none of the false-positive substring collisions that
approach had to guard against there, e.g. "cat" inside "catamaran" —
COCO doesn't even have a "catamaran" class). A direct lookup is simpler
and safer here; use whichever approach actually fits a given method's own
vocabulary, not the same mechanism everywhere for its own sake.

A detected class with no entry below (COCO has 80; this cascade's
categories only need a handful of them) is CATEGORY_OTHER — a normal,
expected outcome, not a gap to keep patching (see categorize()).
"""

from __future__ import annotations

from emil_ml.core.cascade.categories import CATEGORY_ANIMAL, CATEGORY_HUMAN, CATEGORY_OTHER, CATEGORY_VEHICLE

_CATEGORY_BY_COCO_CLASS: dict[str, str] = {
    "person": CATEGORY_HUMAN,
    # Common animals COCO names directly.
    "bird": CATEGORY_ANIMAL,
    "cat": CATEGORY_ANIMAL,
    "dog": CATEGORY_ANIMAL,
    "horse": CATEGORY_ANIMAL,
    "sheep": CATEGORY_ANIMAL,
    "cow": CATEGORY_ANIMAL,
    "elephant": CATEGORY_ANIMAL,
    "bear": CATEGORY_ANIMAL,
    "zebra": CATEGORY_ANIMAL,
    "giraffe": CATEGORY_ANIMAL,
    # Road/common vehicles.
    "bicycle": CATEGORY_VEHICLE,
    "car": CATEGORY_VEHICLE,
    "motorcycle": CATEGORY_VEHICLE,
    "airplane": CATEGORY_VEHICLE,
    "bus": CATEGORY_VEHICLE,
    "train": CATEGORY_VEHICLE,
    "truck": CATEGORY_VEHICLE,
    "boat": CATEGORY_VEHICLE,
    # Every other COCO class (traffic light, backpack, chair, laptop, ...)
    # is deliberately left unmapped -> CATEGORY_OTHER. Nothing in this
    # cascade currently has a specialist for them; adding one later is a
    # one-line addition here plus a specialist_registry.py registration,
    # not a change to categorize() itself.
}


def categorize(coco_class: str) -> str:
    """Map one detected COCO class name (e.g. "person", "car") to a
    coarse category. Unmapped classes are CATEGORY_OTHER — see module
    docstring; not an error.

    Unlike resnet_coarse's categorize(), there is no confidence-based
    CATEGORY_UNCERTAIN case here — that gating happens once, at the
    detection level, in predictor.py (a detection below the component's
    coco_confidence_threshold is dropped before ever reaching this
    function, not passed through and mapped to "uncertain").
    """
    return _CATEGORY_BY_COCO_CLASS.get(coco_class, CATEGORY_OTHER)
