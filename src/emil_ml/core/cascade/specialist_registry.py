"""Maps a specialist NAME to its implementation — the cascade's own
registry_factory.py, structurally identical to that module's (modality,
model_type) -> class dispatch. This module only knows "name -> instance";
which coarse CATEGORY activates which specialist NAME is a separate,
per-component decision — see parse_category_specialists() below and
core/cascade/pipeline.py, which is what actually joins the two together
for one frame.

That split is deliberate, not incidental: a fixed, hardcoded category ->
specialist mapping (this module's own design before per-component
configurability) meant the only way to change "does 'human' trigger face
recognition" was a code change, and there was no way for two different
coco_detector components to make different choices. Now:

- THIS module answers "given a specialist name, get me a working
  instance" — a fixed, code-level registration (like
  core/registry_factory.py's own dispatch dicts), since specialist
  IMPLEMENTATIONS are still code, not configuration.
- A component's own `cascade_category_specialists` setting (JSON, see
  settings.py) answers "for THIS component, which categories are
  activated, and which specialist name handles each" — genuine runtime
  configuration, editable from the Onboard page with no code change,
  covering both "turn face recognition off for human" and "turn a future
  car-model classifier on for vehicle".

A coarse category with no specialist NAME configured for it (or a name
not in _SPECIALIST_FACTORIES below) is a normal, valid, first-class
outcome — the object is still detected and reported (its real COCO class
and coarse category both stay visible; core/cascade/pipeline.py never
throws them away), just with no further identification. Not an error,
not a silent gap.

Every specialist's own heavy-dependency import (torch-based
facenet-pytorch for the face specialist) is deferred to inside its
factory function, not imported at this module's top level — same
reasoning as registry_factory.py's own deferred imports.
"""

from __future__ import annotations

import json
from typing import Callable

from emil_ml.core.cascade.base import BaseSpecialist
from emil_ml.core.cascade.categories import CATEGORY_HUMAN


def _face_recognition_specialist() -> BaseSpecialist:
    from emil_ml.core.cascade.specialists.face.predictor import FaceRecognitionSpecialist

    return FaceRecognitionSpecialist()


# Registered here, not inline above: keeping every entry in one dict (like
# registry_factory.py's _PREDICTOR_FACTORIES) is what makes "is this
# specialist name known" a lookup instead of scattered `if name == ...`
# branches elsewhere in the cascade. Keyed by the specialist's own `.name`
# (BaseSpecialist.name — see base.py), so the key here and the identity
# a policy row is namespaced under (core/cascade/policy_store.py) always
# agree by construction.
_SPECIALIST_FACTORIES: dict[str, Callable[[], BaseSpecialist]] = {
    "face": _face_recognition_specialist,
    # "car_classifier": _car_classifier_specialist,  # future — see module docstring; not built yet
}

# The default per-component `cascade_category_specialists` value (see
# settings.py's DEFAULT_CASCADE_CATEGORY_SPECIALISTS, which a new
# coco_detector component is created with, and what a category missing
# from an existing component's own mapping falls back to). Only "human"
# is activated out of the box — every other category is detect-and-report
# only until an operator deliberately turns on a specialist for it (e.g.
# "vehicle" -> "car_classifier", once that specialist exists).
DEFAULT_CATEGORY_SPECIALISTS: dict[str, str] = {CATEGORY_HUMAN: "face"}


def get_specialist_by_name(specialist_name: str) -> BaseSpecialist | None:
    """Return a fresh specialist instance for `specialist_name`, or None
    if that name isn't registered — a component whose own
    cascade_category_specialists mapping references a specialist that
    doesn't exist (yet) degrades to "no specialist" for that category,
    same as an unconfigured one, rather than raising.
    """
    factory = _SPECIALIST_FACTORIES.get(specialist_name)
    return factory() if factory else None


def available_specialist_names() -> list[str]:
    """Every specialist name a component's cascade_category_specialists
    mapping can reference — what the Onboard page's configuration UI
    offers as choices."""
    return sorted(_SPECIALIST_FACTORIES.keys())


def parse_category_specialists(raw_json: str) -> dict[str, str]:
    """Parse a component's Component.cascade_category_specialists JSON
    into a plain {category: specialist_name} dict — the same
    parse/serialize split core/reporting/machine_context/parameters.py
    established for MachineParameterDef. Empty/missing JSON parses to an
    empty mapping (every category detect-and-report only), not an error.
    """
    if not raw_json:
        return {}
    return json.loads(raw_json)


def serialize_category_specialists(mapping: dict[str, str]) -> str:
    """Inverse of parse_category_specialists() — for writing via
    registry.update_settings()."""
    return json.dumps(mapping)
