"""Common interface every cascade specialist (face recognition today, a
future car-model classifier, ...) implements — the cascade's analog to
core/base.py's BasePredictor/PredictionResult, one layer down.

core/cascade/pipeline.py and core/cascade/specialist_registry.py only
ever talk to BaseSpecialist and SpecialistResult below — never a concrete
specialist class directly. Adding a new specialist means implementing
this interface and registering it in specialist_registry.py; nothing
else in the cascade changes.

Deliberately NOT bound to a Component at construction the way
BasePredictor is: a specialist owns its own resource (the face
specialist's known-individuals embedding database), not a per-component
trained model — the cascade isn't scoped to one Component the way
anomaly inspection is.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

#: A detected object's bounding box in the coarse frame, (x1, y1, x2, y2)
#: in pixel coordinates — whatever the coarse detector's own
#: PredictionResult.details["detections"][i]["box"] carried through
#: core/cascade/pipeline.py. See BaseSpecialist.identify()'s docstring.
BoundingBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class SpecialistResult:
    """Common shape returned by every specialist, regardless of domain.

    `identity_key` is the specialist's own stable lookup key into the
    reaction-policy table (see core/cascade/policy.py) — a person's slug
    today, a car model's slug tomorrow. "unknown" is always a valid,
    first-class `identity_key` (see BaseSpecialist.identify()'s
    docstring) with its own policy row, not a special case threaded
    through the cascade as an exception or None.
    """

    matched: bool
    identity_key: str  # e.g. "unknown", "alice-smith"
    identity_label: str  # human-readable form, e.g. "Unknown", "Alice Smith"
    confidence: float | None
    details: dict[str, Any] = field(default_factory=dict)


class BaseSpecialist(ABC):
    """Identifies a specific instance within a coarse category."""

    #: Namespaces this specialist's identity_keys in the reaction-policy
    #: table (see core/cascade/policy_store.py's (specialist, identity_key)
    #: composite key) — so "unknown" for faces and "unknown" for a future
    #: car specialist never collide.
    name: str

    @abstractmethod
    def identify(self, image: Any, *, box: BoundingBox | None = None) -> SpecialistResult:
        """Identify `image` (already-decoded, e.g. a PIL.Image — whatever
        the coarse stage's modality handler produced) and return a
        SpecialistResult. Must never raise for "nothing found"/"no
        match" — those are `matched=False` results with
        `identity_key="unknown"`, not exceptions; a specialist should
        only raise for a genuine operational failure (e.g. its model
        failed to load).

        `box` is the coarse stage's own bounding box for the specific
        detected object this call is about (see core/cascade/pipeline.py,
        which dispatches one identify() call per detection) — None if the
        coarse stage didn't provide one (e.g. a whole-frame classifier
        rather than a detector). Purely advisory: a specialist decides for
        itself whether/how to use it (e.g. the face specialist crops to
        it before searching for a face) — searching the whole `image`
        instead when `box` is None, or even when it's given, is a valid
        implementation choice, not a contract violation.
        """
        ...
