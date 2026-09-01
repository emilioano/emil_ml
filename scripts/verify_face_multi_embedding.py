"""Verifies the multi-embedding-per-individual redesign of
core/cascade/specialists/face/store.py: registering several photos per
person, matching against the richer representation, adding/removing
individual photos, removing a whole individual (cascade-deletes every
embedding), backward compatibility with a single-photo individual, and
threshold calibration (intra-person spread vs. inter-person distance).

Uses one real photo (skimage's bundled astronaut()) run through several
different PIL transforms (flip, brightness, rotation) to stand in for
"the same person, several photos under different conditions" — genuinely
different embeddings of the same real face, not synthetic vectors, for
everything that needs a REAL face. The one place a synthetic vector is
used (a second "person" for calibration) is called out explicitly, since
this repo has no second bundled real face photo offline.

Run with: python scripts/verify_face_multi_embedding.py
"""

from __future__ import annotations

import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageEnhance
from skimage import data

from emil_ml.config.logging_config import configure_logging
from emil_ml.core.cascade.specialists.face import store as face_store
from emil_ml.core.cascade.specialists.face.predictor import FaceRecognitionSpecialist

PERSON_NAME = "Multi Embedding Test Person"
PERSON_KEY = "multi-embedding-test-person"
OTHER_NAME = "Other Person (synthetic, calibration only)"
OTHER_KEY = "other-person-synthetic-calibration-only"

ALL_PASS = True


def _check(label: str, condition: bool, detail: str = "") -> None:
    global ALL_PASS
    status = "PASS" if condition else "FAIL"
    print(f"  {status}: {label}" + (f" — {detail}" if detail else ""))
    ALL_PASS = ALL_PASS and condition


def _embed(mtcnn, resnet, image: Image.Image) -> list[float] | None:
    face_tensor = mtcnn(image)
    if face_tensor is None:
        return None
    with torch.no_grad():
        return resnet(face_tensor.unsqueeze(0))[0].tolist()


def main() -> None:
    configure_logging()
    face_store.delete_known_individual(PERSON_KEY)
    face_store.delete_known_individual(OTHER_KEY)

    mtcnn = MTCNN(keep_all=False)
    resnet = InceptionResnetV1(pretrained="vggface2").eval()

    base_image = Image.fromarray(data.astronaut()).convert("RGB")
    variant_flipped = base_image.transpose(Image.FLIP_LEFT_RIGHT)
    variant_bright = ImageEnhance.Brightness(base_image).enhance(1.6)
    variant_dark = ImageEnhance.Brightness(base_image).enhance(0.55)
    variant_rotated = base_image.rotate(12, expand=False, fillcolor=(128, 128, 128))
    variant_holdout = base_image.rotate(-8, expand=False, fillcolor=(128, 128, 128))  # never registered — used only to test recognition

    try:
        print("=== 1: register an individual with MULTIPLE photos (different real conditions) ===")
        embeddings_by_variant = {
            "original": _embed(mtcnn, resnet, base_image),
            "flipped": _embed(mtcnn, resnet, variant_flipped),
            "bright": _embed(mtcnn, resnet, variant_bright),
            "dark": _embed(mtcnn, resnet, variant_dark),
        }
        found = {k: v is not None for k, v in embeddings_by_variant.items()}
        print(f"  face detected per variant: {found}")
        _check("MTCNN found a face in every registration variant", all(found.values()), detail=str(found))

        face_store.add_known_individual(PERSON_NAME, embeddings_by_variant["original"], consented=True)
        face_store.add_face_embedding(PERSON_KEY, embeddings_by_variant["flipped"])
        face_store.add_face_embedding(PERSON_KEY, embeddings_by_variant["bright"])
        face_store.add_face_embedding(PERSON_KEY, embeddings_by_variant["dark"])

        individual = face_store.get_by_identity_key(PERSON_KEY)
        _check("individual now has 4 embeddings stored", individual is not None and individual.embedding_count == 4, detail=str(individual))
        print()

        print("=== 2: recognized under a HOLD-OUT condition never used for registration ===")
        holdout_embedding = _embed(mtcnn, resnet, variant_rotated)
        specialist = FaceRecognitionSpecialist()
        # identify() does its own detection internally; feed it the real image directly.
        result = specialist.identify(variant_rotated)
        print(f"  matched={result.matched} identity={result.identity_key} "
              f"distance={result.details.get('distance')}")
        _check(
            "the rotated hold-out photo is recognized as the same person (multi-photo coverage)",
            result.matched and result.identity_key == PERSON_KEY,
        )
        print()

        print("=== 3: add another photo to the already-registered individual ===")
        face_store.add_face_embedding(PERSON_KEY, holdout_embedding)
        individual_after_add = face_store.get_by_identity_key(PERSON_KEY)
        _check("embedding count went from 4 to 5", individual_after_add.embedding_count == 5)
        print()

        print("=== 4: remove ONE photo — person and other photos untouched ===")
        embeddings_before_removal = face_store.list_embeddings_for(PERSON_KEY)
        removed_id = embeddings_before_removal[0].id
        face_store.delete_face_embedding(removed_id)
        individual_after_removal = face_store.get_by_identity_key(PERSON_KEY)
        _check(
            "embedding count went from 5 to 4 after removing one photo",
            individual_after_removal.embedding_count == 4,
        )
        _check("the person themself is still registered", individual_after_removal is not None)
        print()

        print("=== 5: backward compatibility — a person with just ONE photo still works ===")
        single_photo_key = "single-photo-test-person"
        face_store.delete_known_individual(single_photo_key)
        face_store.add_known_individual("Single Photo Test Person", embeddings_by_variant["original"], consented=True)
        single_result = specialist.identify(base_image)
        _check(
            "a single-embedding individual is still matched correctly",
            single_result.matched and single_result.identity_key == single_photo_key,
        )
        face_store.delete_known_individual(single_photo_key)
        print()

        print("=== 6: threshold calibration — intra-person spread vs. inter-person distance ===")
        # A second "person" — no second real bundled photo offline, so a
        # synthetic, deliberately-distant embedding stands in here ONLY for
        # this calibration-math check (not for any face-detection check
        # above, which all use real photos throughout).
        other_embedding = [-x for x in embeddings_by_variant["original"]]
        face_store.add_known_individual(OTHER_NAME, other_embedding, consented=True)

        calibration = face_store.calibration_stats()
        print(f"  intra-person distances (n={len(calibration.intra_person_distances)}): "
              f"{[round(d, 3) for d in calibration.intra_person_distances[:5]]}...")
        print(f"  inter-person distances (n={len(calibration.inter_person_distances)}): "
              f"{[round(d, 3) for d in calibration.inter_person_distances[:5]]}...")
        print(f"  suggested_threshold={calibration.suggested_threshold} separable={calibration.separable}")
        _check("intra-person distances were computed (same person, several photos)", len(calibration.intra_person_distances) > 0)
        _check("inter-person distances were computed (different people)", len(calibration.inter_person_distances) > 0)
        _check(
            "suggested threshold sits strictly between max intra and min inter",
            calibration.suggested_threshold is not None
            and max(calibration.intra_person_distances) < calibration.suggested_threshold < min(calibration.inter_person_distances),
        )
        _check("distributions are reported as separable (synthetic 'other' is far away)", calibration.separable)
        print()

        print("=== 7: unregister the individual entirely — ALL their embeddings are gone, not just one ===")
        face_store.delete_known_individual(PERSON_KEY)
        _check("individual is gone", face_store.get_by_identity_key(PERSON_KEY) is None)
        _check("every one of their embeddings is gone too", face_store.list_embeddings_for(PERSON_KEY) == [])
        print()

        print(f"Overall: {'ALL PASS' if ALL_PASS else 'SOME FAILED — see above'}")
    finally:
        face_store.delete_known_individual(PERSON_KEY)
        face_store.delete_known_individual(OTHER_KEY)
        face_store.delete_known_individual("single-photo-test-person")


if __name__ == "__main__":
    main()
