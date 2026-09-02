"""Drives the Onboard page's cascade UI end-to-end via Streamlit's AppTest
— i.e. exactly the widget interactions a real operator would make, no
direct backend calls — to verify the whole "component -> registered
individual -> policies -> run" flow works entirely from the UI:

1. Create a coco_detector component from the "Analysis method" radio +
   its minimal creation form (no dataset/annotation).
2. Register a consenting individual: upload a photo, confirm the
   detected face preview, tick the (initially blocking) consent
   checkbox, register.
3. Set reaction policies for that individual AND for "unknown".
4. Run the cascade on a test image and confirm the rendered result
   shows the detection, the recognition, and the policy outcome.
5. Remove the registered individual.

Run with: python scripts/verify_onboard_cascade_ui.py
"""

from __future__ import annotations

import io
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from PIL import Image as PILImage
from skimage import data
from streamlit.testing.v1 import AppTest

from emil_ml.config.registry import ComponentRegistry
from emil_ml.core import component_deletion
from emil_ml.core.cascade import policy_store
from emil_ml.core.cascade import specialist_registry
from emil_ml.core.cascade.specialists.face import store as face_store
from emil_ml.utils.paths import slugify

COMPONENT_DISPLAY_NAME = "AppTest Cascade Component"
COMPONENT_NAME = slugify(COMPONENT_DISPLAY_NAME)
PERSON_NAME = "AppTest Alice"
PERSON_KEY = slugify(PERSON_NAME)

ALL_PASS = True


def _check(label: str, condition: bool, detail: str = "") -> None:
    global ALL_PASS
    status = "PASS" if condition else "FAIL"
    print(f"  {status}: {label}" + (f" — {detail}" if detail else ""))
    ALL_PASS = ALL_PASS and condition


def _png_bytes(image: PILImage.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def main() -> None:
    registry = ComponentRegistry()
    # Clean slate: this script is disposable/re-runnable.
    if registry.get(COMPONENT_NAME) is not None:
        registry.soft_delete(COMPONENT_NAME)
        component_deletion.permanently_delete_component(COMPONENT_NAME, registry=registry)
    face_store.delete_known_individual(PERSON_KEY)
    policy_store.delete_policy(COMPONENT_NAME, "face", PERSON_KEY)
    policy_store.delete_policy(COMPONENT_NAME, "face", "unknown")

    try:
        at = AppTest.from_file("app/pages/2_onboard.py", default_timeout=60)
        at.run()
        _check("page loads with no exception", not at.exception)

        print("=== 1: create a coco_detector component from the UI (no Python) ===")
        at.radio[0].set_value("Object & face cascade (COCO detector)")
        at.run()
        name_inputs = [w for w in at.text_input if w.label == "Component display name"]
        _check("creation form's display-name field is present", len(name_inputs) == 1)
        name_inputs[0].set_value(COMPONENT_DISPLAY_NAME)
        submit_buttons = [b for b in at.button if b.label == "Create component"]
        _check("'Create component' button is present", len(submit_buttons) == 1)
        submit_buttons[0].click()
        at.run()
        _check("no exception after creating the component", not at.exception, detail=str(at.exception))

        component = registry.get(COMPONENT_NAME)
        _check("component now exists and is ready", component is not None and component.status == "ready")
        print()

        print("=== 2: register a consenting individual via photo upload (no Python) ===")
        photo_key = f"face_reg_photo_{COMPONENT_NAME}"
        at.file_uploader(key=photo_key).set_value(("alice.png", _png_bytes(PILImage.fromarray(data.astronaut())), "image/png"))
        at.run()
        _check("no exception after uploading the registration photo", not at.exception, detail=str(at.exception))

        # st.image()'s AppTest element wraps an ImageList proto (it supports
        # rendering several images from one call) — the actual caption for a
        # single-image call lives one level down, at proto.imgs[0].caption.
        all_captions = [img.proto.imgs[0].caption for img in at.image if img.proto.imgs]
        preview_captions = [c for c in all_captions if "face(s) detected" in c]
        _check("UI shows a 'face(s) detected' preview caption", len(preview_captions) == 1, detail=str(preview_captions))
        _check("exactly one face was detected (real photo, real MTCNN)", preview_captions and preview_captions[0].startswith("1 face"))

        consent_key = f"face_reg_consent_{COMPONENT_NAME}"
        register_key = f"face_reg_submit_{COMPONENT_NAME}"
        register_button_before = at.button(key=register_key)
        _check("Register button is DISABLED before consent is given", register_button_before.disabled is True)

        name_key = f"face_reg_name_{COMPONENT_NAME}"
        at.text_input(key=name_key).set_value(PERSON_NAME)
        at.run()
        register_button_no_consent = at.button(key=register_key)
        _check("Register button STAYS disabled with a name but no consent tick", register_button_no_consent.disabled is True)

        at.checkbox(key=consent_key).set_value(True)
        at.run()
        register_button_after_consent = at.button(key=register_key)
        _check("Register button becomes ENABLED once consent is explicitly ticked", register_button_after_consent.disabled is False)

        at.button(key=register_key).click()
        at.run()
        _check("no exception after registering", not at.exception, detail=str(at.exception))

        registered = face_store.get_by_identity_key(PERSON_KEY)
        _check(
            "the individual is now in the known-individuals database, consented=True",
            registered is not None and registered.consented is True and registered.embedding_count == 1,
        )
        remove_person_buttons = [b for b in at.button if b.key == f"remove_person_{PERSON_KEY}"]
        _check("the registered individual now appears in the UI's list with a whole-person Remove button", len(remove_person_buttons) == 1)
        print()

        print("=== 2b: add a second photo to the already-registered individual (no Python) ===")
        add_photo_key = f"add_photo_{PERSON_KEY}"
        second_photo = PILImage.fromarray(data.astronaut()).transpose(PILImage.FLIP_LEFT_RIGHT)
        at.file_uploader(key=add_photo_key).set_value(("alice2.png", _png_bytes(second_photo), "image/png"))
        at.run()
        _check("no exception after uploading a second photo", not at.exception, detail=str(at.exception))
        add_photo_button = [b for b in at.button if b.key == f"add_photo_submit_{PERSON_KEY}"]
        _check("'Add this photo' button appears once a face is detected in it", len(add_photo_button) == 1)
        add_photo_button[0].click()
        at.run()
        _check("no exception after adding the second photo", not at.exception, detail=str(at.exception))
        individual_after_add = face_store.get_by_identity_key(PERSON_KEY)
        _check("embedding count is now 2", individual_after_add is not None and individual_after_add.embedding_count == 2)
        print()

        print("=== 2c: remove ONE photo — the person and their other photo are untouched (no Python) ===")
        embeddings_now = face_store.list_embeddings_for(PERSON_KEY)
        remove_embedding_key = f"remove_embedding_{embeddings_now[0].id}"
        remove_embedding_buttons = [b for b in at.button if b.key == remove_embedding_key]
        _check("a per-photo Remove button exists for the first registered photo", len(remove_embedding_buttons) == 1)
        remove_embedding_buttons[0].click()
        at.run()
        _check("no exception after removing one photo", not at.exception, detail=str(at.exception))
        individual_after_remove = face_store.get_by_identity_key(PERSON_KEY)
        _check(
            "embedding count is back to 1, person still registered",
            individual_after_remove is not None and individual_after_remove.embedding_count == 1,
        )
        print()

        print("=== 3: set reaction policies for the individual and for 'unknown' (no Python) ===")
        target_key = f"policy_target_{COMPONENT_NAME}"
        at.selectbox(key=target_key).set_value(PERSON_KEY)
        at.run()
        at.text_input(key=f"policy_label_{COMPONENT_NAME}_{PERSON_KEY}").set_value("approved person")
        at.text_input(key=f"policy_message_{COMPONENT_NAME}_{PERSON_KEY}").set_value("Welcome, Alice.")
        at.multiselect(key=f"policy_actions_{COMPONENT_NAME}_{PERSON_KEY}").set_value(["log", "display"])
        at.run()
        [b for b in at.button if b.key == f"policy_save_{COMPONENT_NAME}_{PERSON_KEY}"][0].click()
        at.run()
        _check("no exception after saving the individual's policy", not at.exception, detail=str(at.exception))

        at.selectbox(key=target_key).set_value("unknown")
        at.run()
        at.multiselect(key=f"policy_actions_{COMPONENT_NAME}_unknown").set_value(["log", "alert", "save_frame"])
        at.selectbox(key=f"policy_priority_{COMPONENT_NAME}_unknown").set_value("high")
        at.run()
        [b for b in at.button if b.key == f"policy_save_{COMPONENT_NAME}_unknown"][0].click()
        at.run()
        _check("no exception after saving the 'unknown' policy", not at.exception, detail=str(at.exception))

        alice_policy = policy_store.get_policy(COMPONENT_NAME, "face", PERSON_KEY)
        unknown_policy = policy_store.get_policy(COMPONENT_NAME, "face", "unknown")
        _check("Alice's policy was actually persisted", alice_policy is not None and alice_policy.label == "approved person")
        _check(
            "unknown's policy was actually persisted with high priority",
            unknown_policy is not None and unknown_policy.priority == "high" and "alert" in unknown_policy.actions,
        )
        print()

        print("=== 4: run the cascade on a test image and see detection + recognition + policy (no Python) ===")
        run_upload_key = f"cascade_run_upload_{COMPONENT_NAME}"
        at.file_uploader(key=run_upload_key).set_value(
            ("test.png", _png_bytes(PILImage.fromarray(data.astronaut())), "image/png")
        )
        at.run()
        run_button = [b for b in at.button if b.key == f"cascade_run_button_{COMPONENT_NAME}"]
        _check("'Run cascade' button is present once an image is uploaded", len(run_button) == 1)
        run_button[0].click()
        at.run()
        _check("no exception after running the cascade", not at.exception, detail=str(at.exception))

        page_text = (
            " ".join(md.value for md in at.markdown)
            + " ".join(c.value for c in at.caption)
            + " ".join(s.value for s in at.success)  # st.success()/st.warning() are their own element
            + " ".join(w.value for w in at.warning)  # types in AppTest, not st.markdown/st.caption
        )
        _check("result shows Alice was recognized", "Recognized" in page_text and "AppTest Alice" in page_text)
        _check("result shows the policy that was actually triggered", "approved person" in page_text and "Welcome, Alice." in page_text)
        print()

        print("=== 4b: category -> specialist activation is configurable from the UI, not fixed (no Python) ===")
        category_key = f"category_specialist_{COMPONENT_NAME}_human"
        human_selectbox = at.selectbox(key=category_key)
        _check("'human' category selectbox defaults to 'face' (DEFAULT_CATEGORY_SPECIALISTS)", human_selectbox.value == "face")

        no_specialist_option = "(none — detect & report only)"  # must match NO_SPECIALIST_OPTION in app/pages/2_onboard.py
        at.selectbox(key=category_key).set_value(no_specialist_option)
        at.run()
        save_config_buttons = [b for b in at.button if b.key == f"save_category_specialists_{COMPONENT_NAME}"]
        _check("'Save category -> specialist configuration' button is present", len(save_config_buttons) == 1)
        save_config_buttons[0].click()
        at.run()
        _check("no exception after saving the disabled configuration", not at.exception, detail=str(at.exception))

        disabled_component = registry.get(COMPONENT_NAME)
        _check(
            "component's cascade_category_specialists no longer activates 'human'",
            "human" not in specialist_registry.parse_category_specialists(disabled_component.cascade_category_specialists),
        )

        # Re-run with the same already-uploaded test image, now that identification is off for 'human'.
        [b for b in at.button if b.key == f"cascade_run_button_{COMPONENT_NAME}"][0].click()
        at.run()
        _check("no exception after re-running with activation off", not at.exception, detail=str(at.exception))
        page_text_disabled = " ".join(c.value for c in at.caption)
        _check(
            "result now shows 'no specialist activated' instead of a recognition",
            "No specialist activated for this category" in page_text_disabled,
        )
        print()

        print("=== 4c: turning it back on re-activates identification, still no code (no Python) ===")
        at.selectbox(key=category_key).set_value("face")
        at.run()
        [b for b in at.button if b.key == f"save_category_specialists_{COMPONENT_NAME}"][0].click()
        at.run()
        [b for b in at.button if b.key == f"cascade_run_button_{COMPONENT_NAME}"][0].click()
        at.run()
        _check("no exception after re-enabling and re-running", not at.exception, detail=str(at.exception))
        page_text_reenabled = " ".join(s.value for s in at.success)
        _check(
            "Alice is recognized again after re-enabling",
            "AppTest Alice" in page_text_reenabled and "Recognized" in page_text_reenabled,
        )
        print()

        print("=== 5: unregister the individual entirely — all photos gone (no Python) ===")
        [b for b in at.button if b.key == f"remove_person_{PERSON_KEY}"][0].click()
        at.run()
        _check("no exception after removing", not at.exception, detail=str(at.exception))
        _check("the individual is gone from the known-individuals database", face_store.get_by_identity_key(PERSON_KEY) is None)
        _check("every one of their embeddings is gone too", face_store.list_embeddings_for(PERSON_KEY) == [])
        print()

        print(f"Overall: {'ALL PASS' if ALL_PASS else 'SOME FAILED — see above'}")
    finally:
        face_store.delete_known_individual(PERSON_KEY)
        policy_store.delete_policy(COMPONENT_NAME, "face", PERSON_KEY)
        policy_store.delete_policy(COMPONENT_NAME, "face", "unknown")
        if registry.get(COMPONENT_NAME) is not None:
            registry.soft_delete(COMPONENT_NAME)
            component_deletion.permanently_delete_component(COMPONENT_NAME, registry=registry)


if __name__ == "__main__":
    main()
