"""The coarse-category vocabulary every Step-1 method must speak — the
shared contract that keeps core/cascade/specialist_registry.py's
category -> specialist dispatch (Step 2) working identically regardless
of which coarse method is currently registered as Step 1.

Factored out on its own (rather than left inside one coarse method's own
module, which is where it originally lived when
core/classification/resnet_coarse was the only coarse method) specifically
because there are now two coarse methods that both need to produce
exactly these same category strings: resnet_coarse's ImageNet mapping and
core/detection/yolo_coco's COCO mapping. A single source of truth here
means the two can never silently drift on the literal string value of
"human", which would otherwise break Step 2 dispatch for one of them
without either module's own tests noticing (each only checks its own
mapping in isolation).
"""

from __future__ import annotations

CATEGORY_ANIMAL = "animal"
CATEGORY_VEHICLE = "vehicle"
CATEGORY_HUMAN = "human"
CATEGORY_OTHER = "other"
CATEGORY_UNCERTAIN = "uncertain"  # confidence below the coarse method's own threshold
