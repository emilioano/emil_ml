"""Verifies Steps 2-4 of the cascade framework (category->specialist
dispatch, face-recognition matching/unknown-fallback, reaction-policy
lookup/execution) plus the full run_cascade() orchestrator end-to-end,
now on COCO-YOLO as Step 1 (see core/detection/yolo_coco) — real "human"
detections at last, not the old ImageNet-1k classifier's ~26%-confidence
guess (see scripts/verify_cascade_step1_resnet.py, still valid, still
demonstrates that old limitation directly).

Step 1 in isolation is verified separately in
scripts/verify_cascade_step1_coco.py — this script focuses on everything
downstream of a coarse detection, using real face detection/embedding/
matching throughout (skimage's bundled real photos, no network access
needed). Steps 2-4's own logic (checks 1-9) is UNCHANGED from before
COCO-YOLO replaced the coarse stage — only what feeds them changed; see
core/cascade/__init__.py.
"""

from __future__ import annotations

import shutil
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from PIL import Image
from skimage import data

from emil_ml.config.logging_config import configure_logging
from emil_ml.config.registry import ComponentRegistry
from emil_ml.config.settings import CASCADE_SAVED_FRAMES_DIR
from emil_ml.core.cascade import pipeline, policy_executor, policy_store, specialist_registry
from emil_ml.core.cascade.categories import CATEGORY_ANIMAL, CATEGORY_HUMAN
from emil_ml.core.cascade.specialists.face import store as face_store
from emil_ml.core.cascade.specialists.face.predictor import FaceRecognitionSpecialist
from emil_ml.training import onboard

COMPONENT_DISPLAY_NAME = "Cascade Full Test Component"
TEST_PERSON_NAME = "Cascade Test Person"
TEST_PERSON_KEY = "cascade-test-person"


def main() -> None:
    configure_logging()
    all_pass = True

    astronaut_image = Image.fromarray(data.astronaut()).convert("RGB")
    cat_image = Image.fromarray(data.chelsea()).convert("RGB")

    # Clean slate: this script is disposable/re-runnable.
    face_store.delete_known_individual(TEST_PERSON_KEY)
    policy_store.delete_policy("face", TEST_PERSON_KEY)
    policy_store.delete_policy("face", "unknown")

    try:
        print("=== 1: specialist_registry is name-keyed; DEFAULT_CATEGORY_SPECIALISTS activates only 'human' ===")
        face_specialist = specialist_registry.get_specialist_by_name("face")
        unknown_name_specialist = specialist_registry.get_specialist_by_name("car_classifier")
        default_mapping = specialist_registry.DEFAULT_CATEGORY_SPECIALISTS
        ok1 = (
            isinstance(face_specialist, FaceRecognitionSpecialist)
            and unknown_name_specialist is None
            and "face" in specialist_registry.available_specialist_names()
            and default_mapping.get(CATEGORY_HUMAN) == "face"
            and CATEGORY_ANIMAL not in default_mapping
            and "vehicle" not in default_mapping
        )
        print(f"  get_specialist_by_name('face') -> {type(face_specialist).__name__}, "
              f"('car_classifier') -> {unknown_name_specialist}")
        print(f"  DEFAULT_CATEGORY_SPECIALISTS = {default_mapping}")
        print(f"-> {'PASS' if ok1 else 'FAIL'}: an unregistered specialist name degrades to None gracefully; "
              f"only 'human' is activated by default.")
        all_pass &= ok1
        print()

        print("=== 2: consent is structurally enforced — consented=False is rejected ===")
        ok2 = False
        try:
            face_store.add_known_individual("Should Not Be Added", [0.0] * 512, consented=False)
        except ValueError as exc:
            ok2 = True
            print(f"  raised ValueError as expected: {exc}")
        print(f"-> {'PASS' if ok2 else 'FAIL'}")
        all_pass &= ok2
        print()

        print("=== 3: FaceRecognitionSpecialist detects+embeds a real face, matches a consenting individual ===")
        specialist = FaceRecognitionSpecialist()
        # Real detect+embed on a real photo, to register the known individual — no synthetic vectors here.
        pre_result = specialist.identify(astronaut_image)
        ok3a = pre_result.matched is False and pre_result.identity_key == "unknown"
        print(f"  before registering anyone: matched={pre_result.matched} "
              f"reason={pre_result.details.get('reason')}")

        # Recover the real embedding the specialist just computed, by re-running detection+embedding
        # directly (mirrors what predictor.py does internally) so we can register it as a known individual.
        import torch
        from facenet_pytorch import MTCNN, InceptionResnetV1

        mtcnn = MTCNN(keep_all=False)
        resnet = InceptionResnetV1(pretrained="vggface2").eval()
        face_tensor = mtcnn(astronaut_image)
        with torch.no_grad():
            real_embedding = resnet(face_tensor.unsqueeze(0))[0].tolist()

        face_store.add_known_individual(TEST_PERSON_NAME, real_embedding, consented=True)
        post_result = specialist.identify(astronaut_image)
        ok3b = (
            post_result.matched is True
            and post_result.identity_key == TEST_PERSON_KEY
            and post_result.identity_label == TEST_PERSON_NAME
            and post_result.details["distance"] < 0.01  # same image -> near-zero self-distance
        )
        print(f"  after registering '{TEST_PERSON_NAME}': matched={post_result.matched} "
              f"identity={post_result.identity_key} distance={post_result.details.get('distance'):.6f}")
        ok3 = ok3a and ok3b
        print(f"-> {'PASS' if ok3 else 'FAIL'}")
        all_pass &= ok3
        print()

        print("=== 4: a face that doesn't match anyone falls back to 'unknown' (not a crash/misassignment) ===")
        far_embedding = [-x for x in real_embedding]  # deliberately distant from the real one
        no_match = face_store.find_best_match(far_embedding, threshold=0.9)
        ok4 = no_match is None
        print(f"  distant synthetic embedding match result: {no_match}")
        print(f"-> {'PASS' if ok4 else 'FAIL'}")
        all_pass &= ok4
        print()

        print("=== 5: a frame with no face at all (real cat photo) also falls back to 'unknown' ===")
        cat_result = specialist.identify(cat_image)
        ok5 = cat_result.matched is False and cat_result.details.get("reason") == "no_face_detected"
        print(f"  matched={cat_result.matched} reason={cat_result.details.get('reason')}")
        print(f"-> {'PASS' if ok5 else 'FAIL'}")
        all_pass &= ok5
        print()

        print("=== 6: a new person is added purely via configuration (add_known_individual), no retraining ===")
        roster = {p.identity_key for p in face_store.list_known_individuals()}
        ok6 = TEST_PERSON_KEY in roster
        print(f"  roster contains '{TEST_PERSON_KEY}': {ok6}")
        print(f"-> {'PASS' if ok6 else 'FAIL'}")
        all_pass &= ok6
        print()

        print("=== 7: reaction policy — unconfigured identity falls back to a safe default (log-only) ===")
        fallback_result = policy_executor.execute_policy("face", "someone-with-no-policy-yet")
        ok7 = fallback_result.executed_actions == ("log",) and fallback_result.saved_frame_path is None
        print(f"  executed_actions={fallback_result.executed_actions} label={fallback_result.policy.label}")
        print(f"-> {'PASS' if ok7 else 'FAIL'}")
        all_pass &= ok7
        print()

        print("=== 8: a configured identity's policy executes its OWN distinct actions, incl. save_frame ===")
        policy_store.upsert_policy(
            "face", TEST_PERSON_KEY,
            label="approved person", message="Welcome back!", actions=["log", "display", "save_frame"],
            priority="normal",
        )
        policy_store.upsert_policy(
            "face", "unknown",
            label="unknown", message="Unrecognized individual detected.", actions=["log", "alert", "save_frame"],
            priority="high",
        )
        observed_actions: list[str] = []
        known_policy_result = policy_executor.execute_policy(
            "face", TEST_PERSON_KEY, image=astronaut_image, on_action=lambda action, _policy: observed_actions.append(action)
        )
        unknown_policy_result = policy_executor.execute_policy("face", "unknown", image=cat_image)
        ok8 = (
            known_policy_result.policy.label == "approved person"
            and known_policy_result.executed_actions == ("log", "display", "save_frame")
            and known_policy_result.saved_frame_path is not None
            and known_policy_result.saved_frame_path.exists()
            and observed_actions == ["log", "display", "save_frame"]
            and unknown_policy_result.policy.label == "unknown"
            and unknown_policy_result.policy.priority == "high"
            and unknown_policy_result.executed_actions == ("log", "alert", "save_frame")
            and unknown_policy_result.saved_frame_path != known_policy_result.saved_frame_path
        )
        print(f"  known person policy: label={known_policy_result.policy.label} "
              f"actions={known_policy_result.executed_actions} saved={known_policy_result.saved_frame_path}")
        print(f"  unknown policy: label={unknown_policy_result.policy.label} "
              f"priority={unknown_policy_result.policy.priority} actions={unknown_policy_result.executed_actions}")
        print(f"-> {'PASS' if ok8 else 'FAIL'}: known-person and unknown policies are genuinely distinct, "
              f"identity-agnostic lookup (same execute_policy(), different configured data).")
        all_pass &= ok8
        print()

        print("=== 9: upsert_policy() rejects an invalid action instead of silently accepting it ===")
        ok9 = False
        try:
            policy_store.upsert_policy("face", "bad-actions-test", label="x", message="x", actions=["teleport"])
        except ValueError as exc:
            ok9 = True
            print(f"  raised ValueError as expected: {exc}")
        print(f"-> {'PASS' if ok9 else 'FAIL'}")
        all_pass &= ok9
        print()

        registry = ComponentRegistry()
        coarse_component = onboard.create_component(
            COMPONENT_DISPLAY_NAME, model_type="coco_detector", registry=registry
        )
        onboard.train_component(coarse_component.name, registry=registry)

        print("=== 10: run_cascade() end-to-end — real 'animal' detection, NO specialist registered (graceful stop) ===")
        cascade_result_animal = pipeline.run_cascade(cat_image, coarse_component.name, registry=registry)
        ok10 = (
            len(cascade_result_animal.objects) >= 1
            and cascade_result_animal.objects[0].category == CATEGORY_ANIMAL
            and cascade_result_animal.objects[0].label == "cat"
            and cascade_result_animal.objects[0].specialist_result is None
            and cascade_result_animal.objects[0].policy_result is None
        )
        print(f"  objects: {[(o.label, o.category, o.specialist_result) for o in cascade_result_animal.objects]}")
        print(f"-> {'PASS' if ok10 else 'FAIL'}: cascade stops cleanly at the coarse detection, not an error.")
        all_pass &= ok10
        print()

        print("=== 11: run_cascade() end-to-end — real 'human' detection, NO monkey-patching needed anymore ===")
        print("        (COCO-YOLO genuinely detects 'person'; this is the exact branch ImageNet-1k could never")
        print("        reach — see verify_cascade_step1_resnet.py). Full 4-step chain fires through run_cascade().")
        full_result = pipeline.run_cascade(astronaut_image, coarse_component.name, registry=registry)
        human_objects = [o for o in full_result.objects if o.category == CATEGORY_HUMAN]
        ok11 = (
            len(human_objects) == 1
            and human_objects[0].label == "person"
            and human_objects[0].box is not None
            and human_objects[0].specialist_result is not None
            and human_objects[0].specialist_result.matched is True
            and human_objects[0].specialist_result.identity_key == TEST_PERSON_KEY
            and human_objects[0].policy_result is not None
            and human_objects[0].policy_result.policy.label == "approved person"
        )
        print(f"  all detected objects: {[(o.label, o.category, o.box is not None) for o in full_result.objects]}")
        print(f"  human object: box={human_objects[0].box if human_objects else None}")
        if human_objects:
            print(f"  specialist matched={human_objects[0].specialist_result.matched} "
                  f"identity={human_objects[0].specialist_result.identity_key}")
            print(f"  policy executed: label={human_objects[0].policy_result.policy.label} "
                  f"actions={human_objects[0].policy_result.executed_actions}")
        print(f"-> {'PASS' if ok11 else 'FAIL'}")
        all_pass &= ok11
        print()

        print("=== 12: bounding box is genuinely propagated to the specialist (not just present on the object) ===")
        ok12 = (
            len(human_objects) == 1
            and human_objects[0].specialist_result.details.get("searched_box") == human_objects[0].box
        )
        print(f"  object.box={human_objects[0].box if human_objects else None}")
        print(f"  specialist_result.details['searched_box']="
              f"{human_objects[0].specialist_result.details.get('searched_box') if human_objects else None}")
        print(f"-> {'PASS' if ok12 else 'FAIL'}: the specialist actually received and used the same box, "
              f"not the whole frame.")
        all_pass &= ok12
        print()

        print("=== 13: multiple objects in one frame — EVERY detection is dispatched, not just the top one ===")
        composite = Image.new("RGB", (astronaut_image.width + cat_image.width, max(astronaut_image.height, cat_image.height)))
        composite.paste(astronaut_image, (0, 0))
        composite.paste(cat_image, (astronaut_image.width, 0))
        composite_result = pipeline.run_cascade(composite, coarse_component.name, registry=registry)
        composite_human = [o for o in composite_result.objects if o.category == CATEGORY_HUMAN]
        composite_animal = [o for o in composite_result.objects if o.category == CATEGORY_ANIMAL]
        ok13 = (
            len(composite_result.objects) >= 2
            and len(composite_human) >= 1
            and composite_human[0].specialist_result is not None
            and composite_human[0].specialist_result.matched is True
            and len(composite_animal) >= 1
            and composite_animal[0].specialist_result is None  # no specialist for "animal" — untouched, not an error
        )
        print(f"  composite frame objects: "
              f"{[(o.label, o.category, o.specialist_result is not None) for o in composite_result.objects]}")
        print(f"-> {'PASS' if ok13 else 'FAIL'}: both the person (specialist ran) and the cat (no specialist, "
              f"left alone) came back in one run_cascade() call — real iteration, not 'first detection only'.")
        all_pass &= ok13
        print()

        print("=== 14: full object vocabulary stays visible — an 'other'-category object keeps its real COCO class ===")
        other_objects = [o for o in full_result.objects if o.category == "other"]
        ok14 = len(other_objects) >= 1 and all(o.label and o.label != "other" for o in other_objects)
        print(f"  'other'-category objects: {[(o.label, o.category) for o in other_objects]}")
        print(f"-> {'PASS' if ok14 else 'FAIL'}: e.g. a 'sports ball' detection reports as "
              f"label='sports ball', category='other' — never collapsed to just 'other'.")
        all_pass &= ok14
        print()

        print("=== 15: category -> specialist activation is genuinely per-component configurable ===")
        registry.update_settings(coarse_component.name, cascade_category_specialists="{}")
        disabled_component = registry.get(coarse_component.name)
        disabled_result = pipeline.run_cascade(astronaut_image, disabled_component.name, registry=registry)
        disabled_human = [o for o in disabled_result.objects if o.category == CATEGORY_HUMAN]
        ok15a = (
            len(disabled_human) == 1
            and disabled_human[0].label == "person"  # still detected and reported...
            and disabled_human[0].specialist_result is None  # ...but no longer identified
        )
        print(f"  with cascade_category_specialists='{{}}': human object still detected "
              f"(label={disabled_human[0].label if disabled_human else None}), "
              f"specialist_result={disabled_human[0].specialist_result if disabled_human else 'N/A'}")
        print(f"-> {'PASS' if ok15a else 'FAIL'}: turning the activation off stops identification, "
              f"not detection/reporting.")
        all_pass &= ok15a

        registry.update_settings(
            coarse_component.name,
            cascade_category_specialists=specialist_registry.serialize_category_specialists({CATEGORY_HUMAN: "face"}),
        )
        reenabled_result = pipeline.run_cascade(astronaut_image, coarse_component.name, registry=registry)
        reenabled_human = [o for o in reenabled_result.objects if o.category == CATEGORY_HUMAN]
        ok15b = (
            len(reenabled_human) == 1
            and reenabled_human[0].specialist_result is not None
            and reenabled_human[0].specialist_result.matched is True
            and reenabled_human[0].specialist_result.identity_key == TEST_PERSON_KEY
        )
        print(f"  re-enabled: specialist matched={reenabled_human[0].specialist_result.matched if reenabled_human else None} "
              f"identity={reenabled_human[0].specialist_result.identity_key if reenabled_human else None}")
        print(f"-> {'PASS' if ok15b else 'FAIL'}: turning it back on re-activates identification, no code change either time.")
        all_pass &= ok15b
        print()

        print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED — see above'}")
    finally:
        face_store.delete_known_individual(TEST_PERSON_KEY)
        policy_store.delete_policy("face", TEST_PERSON_KEY)
        policy_store.delete_policy("face", "unknown")
        policy_store.delete_policy("face", "someone-with-no-policy-yet")
        policy_store.delete_policy("face", "bad-actions-test")
        registry = ComponentRegistry()
        if registry.get(COMPONENT_DISPLAY_NAME.lower().replace(" ", "-")) or registry.get(
            "cascade-full-test-component"
        ):
            name = "cascade-full-test-component"
            if registry.get(name) is not None:
                registry.soft_delete(name)
                from emil_ml.core import component_deletion

                component_deletion.permanently_delete_component(name, registry=registry)
        shutil.rmtree(CASCADE_SAVED_FRAMES_DIR / "face" / TEST_PERSON_KEY, ignore_errors=True)
        shutil.rmtree(CASCADE_SAVED_FRAMES_DIR / "face" / "unknown", ignore_errors=True)


if __name__ == "__main__":
    main()
