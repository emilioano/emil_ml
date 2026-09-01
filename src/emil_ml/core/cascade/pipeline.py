"""Single entry point for running the cascade: raw frame -> coarse
detections -> (per detected object) specialist identification ->
reaction-policy execution. The cascade's analog to pipeline/inspect.py,
but for multi-stage, multi-object identification instead of a
single-model anomaly verdict — deliberately a separate entry point, not a
mode of pipeline.inspect(): that function's threshold_override rewrite
and RAG report generation are specific to the approved/failed
anomaly-inspection use case and don't apply here.

Contains NO branching on coarse category, specialist, or identity
anywhere — dispatch at every stage is data:
- Step 1 (coarse detection) comes from whichever (modality, model_type)
  the caller's `coarse_component_name` points at, resolved through
  core.registry_factory exactly like pipeline.inspect() does — this
  function never knows or cares that it's COCO-YOLO today (it was an
  ImageNet-1k classifier before; see
  core/detection/yolo_coco/__init__.py for why that changed). The one
  thing this function DOES require of Step 1's PredictionResult is a
  specific shape: `details["detections"]` — a list of dicts, each with
  "class", "category", "confidence", and "box" — see
  core/detection/yolo_coco/predictor.py for the reference
  implementation. Any coarse method that produces that shape plugs in
  here unchanged.
- Step 2 (category -> specialist) comes from the coarse component's OWN
  `cascade_category_specialists` setting (see settings.py and
  core/cascade/specialist_registry.py's parse_category_specialists()) —
  per-component configuration, not a fixed mapping in this file or in
  specialist_registry.py. A category absent from that mapping (or mapped
  to a specialist name that isn't registered) has no specialist run for
  it — first-class, not an error; the detection is still reported with
  its real COCO class and coarse category intact (see DetectedObject).
- Step 3 (specialist -> identity) is whatever the specialist itself
  returns, given the detected object's own bounding box.
- Step 4 (identity -> reaction) comes from
  core/cascade/policy_executor.py's execute_policy().

MULTIPLE OBJECTS PER FRAME — a deliberate, explicit choice, not an
accident of whichever detection happened to be examined first: a
COCO-YOLO frame can contain several objects (two people; a person and a
car), and run_cascade() dispatches Steps 2-4 for EVERY detection that
cleared the coarse detector's own confidence floor, not just the top one.
Silently only handling the highest-confidence detection would silently
drop a second person from a two-person frame — the exact kind of
behavior this module's own "no hidden branching" principle exists to
prevent. See CascadeResult.objects.

Adding a new coarse category, a new specialist, or a new identity's
policy never touches this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from emil_ml.config.registry import Component, ComponentRegistry
from emil_ml.core import registry_factory
from emil_ml.core.base import PredictionResult
from emil_ml.core.cascade import policy_executor, specialist_registry
from emil_ml.core.cascade.base import BoundingBox, SpecialistResult
from emil_ml.core.cascade.policy import ReactionPolicy
from emil_ml.core.cascade.policy_executor import PolicyExecutionResult


@dataclass(frozen=True)
class DetectedObject:
    """One coarse detection plus whatever the cascade did about it."""

    label: str  # the coarse detector's own raw class name, e.g. "person" (COCO) — method-specific, not part of the shared category vocabulary
    category: str  # mapped coarse category, e.g. "human" — see core/cascade/categories.py
    confidence: float
    box: BoundingBox | None
    specialist_result: SpecialistResult | None  # None when no specialist is registered for this category
    policy_result: PolicyExecutionResult | None  # None when specialist_result is None — nothing to react to


@dataclass(frozen=True)
class CascadeResult:
    """Everything the cascade produced for one frame."""

    coarse: PredictionResult  # Step 1's raw result, kept for debugging/traceability
    objects: list[DetectedObject]  # one entry per coarse detection, in the order Step 1 returned them


def run_cascade(
    raw_input: Any,
    coarse_component_name: str,
    *,
    registry: ComponentRegistry | None = None,
    on_action: Callable[[str, ReactionPolicy], None] | None = None,
) -> CascadeResult:
    """Run the full cascade for one frame against `coarse_component_name`
    (a Component with any model_type registered in registry_factory whose
    predictor produces the `details["detections"]` shape described in this
    module's own docstring — model_type='coco_detector', see
    core/detection/yolo_coco, is the reference implementation and the
    cascade's current default coarse method).

    `raw_input` is whatever the coarse component's modality expects (for
    "image": a path, bytes, PIL.Image, or ndarray) — loading/decoding is
    delegated to the modality handler exactly like pipeline.inspect().
    """
    registry = registry or ComponentRegistry()
    component = registry.get(coarse_component_name)
    if component is None:
        raise KeyError(f"No component named {coarse_component_name!r}")
    if component.status != "ready":
        raise ValueError(
            f"Component {coarse_component_name!r} is not ready for inspection (status={component.status!r})"
        )

    image = _load_frame(component, raw_input)

    coarse_predictor = registry_factory.get_predictor(component.modality, component.model_type, component)
    coarse_result = coarse_predictor.predict(image)

    category_specialists = specialist_registry.parse_category_specialists(component.cascade_category_specialists)
    objects = [
        _process_detection(image, detection, category_specialists, on_action=on_action)
        for detection in coarse_result.details.get("detections", [])
    ]
    return CascadeResult(coarse=coarse_result, objects=objects)


def _process_detection(
    image: Any,
    detection: dict[str, Any],
    category_specialists: dict[str, str],
    *,
    on_action: Callable[[str, ReactionPolicy], None] | None,
) -> DetectedObject:
    """Steps 2-4 for one coarse detection: dispatch a specialist (if the
    component's own config activates one for its category), identify
    within its box, react per policy."""
    category = detection["category"]
    box = tuple(detection["box"]) if detection.get("box") is not None else None

    specialist_name = category_specialists.get(category)
    specialist = specialist_registry.get_specialist_by_name(specialist_name) if specialist_name else None
    if specialist is None:
        return DetectedObject(
            label=detection["class"],
            category=category,
            confidence=detection["confidence"],
            box=box,
            specialist_result=None,
            policy_result=None,
        )

    specialist_result = specialist.identify(image, box=box)
    policy_result = policy_executor.execute_policy(
        specialist.name, specialist_result.identity_key, image=image, on_action=on_action
    )
    return DetectedObject(
        label=detection["class"],
        category=category,
        confidence=detection["confidence"],
        box=box,
        specialist_result=specialist_result,
        policy_result=policy_result,
    )


def _load_frame(component: Component, raw_input: Any) -> Any:
    handler = registry_factory.get_modality_handler(component.modality, component)
    return handler.load(raw_input)
