"""Generic cascade framework: coarse detection -> specialist
identification -> reaction policy, for streamed image/video frames.

Four layers, each dispatched through its own small, swappable mapping —
the same "generic code, specific config" split core/registry_factory.py
already established for (modality, model_type) -> predictor/trainer, one
level up:

1. **Coarse detection** — any (modality, model_type) registered in
   core/registry_factory.py can serve as the coarse stage, as long as its
   PredictionResult carries a `details["detections"]` list (see
   core/cascade/pipeline.py's own docstring for the exact shape). The
   current default is model_type='coco_detector', a COCO-pretrained YOLO
   (core/detection/yolo_coco) — chosen after a whole-frame ImageNet-1k
   classifier (core/classification/resnet_coarse, still available, no
   longer wired here) turned out to have no reliable "person" class,
   permanently starving the person -> face-recognition branch below.
   COCO-YOLO finds POSSIBLY SEVERAL objects per frame, each with its own
   class, bounding box, and confidence, mapped to a coarse category
   string (e.g. "human", "animal", "vehicle", "other", "uncertain").

2. **Category -> specialist dispatch** (specialist_registry.py) — each
   detected object's coarse category looks up a specialist
   implementation. A category with no registered specialist is a normal,
   valid outcome: that detection is simply left un-reacted-to, not an
   error.

3. **Specialist identification** (base.py's BaseSpecialist; the first
   implementation is specialists/face, embedding-based face recognition
   against a consenting-individuals-only database). Always returns a
   SpecialistResult, matched or not — "unknown"/"no match" is data
   flowing through the cascade, never an exception. Receives the
   detected object's own bounding box (optional, advisory) so it can
   narrow its search to that region instead of the whole frame.

4. **Reaction policy** (policy.py / policy_store.py / policy_executor.py)
   — the identified identity_key looks up a structured reaction policy
   (label, message, actions) and policy_executor.py performs it.
   Identity-agnostic: execute_policy() never references a specific
   person (or, for a future specialist, a specific car model) by name;
   all of that knowledge lives in the policy table, mirroring
   core/reporting/machine_context/analyzer.py's parameter-agnostic
   design for machine readings.

pipeline.py's run_cascade() is the single entry point that walks all four
layers for one frame, once per detected object (a frame can contain
several — two people, or a person and a car — see pipeline.py's own
docstring for why iterating every detection is the deliberate choice). It
contains no branching on category, specialist, or identity anywhere.

Nothing here assumes the first specialist (faces) is the only kind that
will ever exist: BaseSpecialist/SpecialistResult and the reaction-policy
table are both keyed generically (by coarse category and by
(specialist_name, identity_key) respectively), so a second specialist
(e.g. a car-model classifier for the "vehicle" category) is a new
registration in specialist_registry.py plus its own policy rows — no
change to pipeline.py, base.py, policy.py, or policy_executor.py.
"""
