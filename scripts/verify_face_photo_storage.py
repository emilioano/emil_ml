"""Verifies the (deliberate, confirmed-with-the-project-owner) reversal of
the face store's original embeddings-only policy: registered photos are now
ALSO saved to disk, downscaled, under settings.KNOWN_INDIVIDUAL_PHOTOS_DIR,
one file per embedding named by that embedding's own row id.

Covers: registering with a photo, adding another photo to an existing
individual, backward compatibility (an embedding added with photo=None still
round-trips with photo_path=None, exactly like every embedding added before
this feature existed), removing ONE photo also removes its file (others
untouched), and — the consent-completeness guarantee this whole module's
privacy framing depends on — removing a whole individual removes EVERY one
of their photo files, not just the DB rows.

Uses skimage's bundled astronaut() photo (a real, detectable face), same
convention scripts/verify_cascade_full.py already established.

Run with: python scripts/verify_face_photo_storage.py
"""

from __future__ import annotations

import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageEnhance
from skimage import data

from emil_ml.config.logging_config import configure_logging
from emil_ml.config.settings import KNOWN_INDIVIDUAL_PHOTOS_DIR
from emil_ml.core.cascade.specialists.face import store as face_store

PERSON_NAME = "Photo Storage Test Person"
PERSON_KEY = "photo-storage-test-person"

ALL_PASS = True


def _check(label: str, condition: bool, detail: str = "") -> None:
    global ALL_PASS
    status = "PASS" if condition else "FAIL"
    print(f"  {status}: {label}" + (f" — {detail}" if detail else ""))
    ALL_PASS = ALL_PASS and condition


def main() -> None:
    configure_logging()
    face_store.delete_known_individual(PERSON_KEY)

    mtcnn = MTCNN(keep_all=False)
    resnet = InceptionResnetV1(pretrained="vggface2").eval()

    def embed(image: Image.Image) -> list[float]:
        face_tensor = mtcnn(image)
        with torch.no_grad():
            return resnet(face_tensor.unsqueeze(0))[0].tolist()

    base_image = Image.fromarray(data.astronaut()).convert("RGB")
    variant_image = ImageEnhance.Brightness(base_image).enhance(1.5)

    try:
        print("=== 1: register with a photo — embedding row gets a real photo_path ===")
        individual = face_store.add_known_individual(
            PERSON_NAME, embed(base_image), consented=True, photo=base_image
        )
        embeddings = face_store.list_embeddings_for(PERSON_KEY)
        _check("exactly one embedding registered", len(embeddings) == 1, detail=str(len(embeddings)))
        first_embedding = embeddings[0]
        _check("photo_path is set", bool(first_embedding.photo_path), detail=str(first_embedding.photo_path))
        _check(
            "photo file actually exists on disk",
            bool(first_embedding.photo_path) and __import__("pathlib").Path(first_embedding.photo_path).exists(),
        )
        _check(
            "photo file is named after its own embedding id",
            first_embedding.photo_path is not None and first_embedding.photo_path.endswith(f"{first_embedding.id}.png"),
            detail=first_embedding.photo_path or "",
        )
        with Image.open(first_embedding.photo_path) as im:
            im.verify()
        _check("photo file is a valid image", True)
        print()

        print("=== 2: add another photo, WITH a photo this time — second file created ===")
        second_embedding = face_store.add_face_embedding(PERSON_KEY, embed(variant_image), photo=variant_image)
        _check("second embedding has its own photo_path", bool(second_embedding.photo_path))
        _check(
            "second photo's path differs from the first (own file, not overwritten)",
            second_embedding.photo_path != first_embedding.photo_path,
        )
        embeddings_after = face_store.list_embeddings_for(PERSON_KEY)
        _check("two embeddings now registered", len(embeddings_after) == 2, detail=str(len(embeddings_after)))
        print()

        print("=== 3: backward compatibility — add_face_embedding() with NO photo still works, photo_path=None ===")
        third_embedding = face_store.add_face_embedding(PERSON_KEY, embed(base_image))  # photo omitted entirely
        _check("photo_path is None when no photo was given", third_embedding.photo_path is None)
        print()

        print("=== 4: removing ONE photo removes its file, leaves the others untouched ===")
        path_to_be_removed = first_embedding.photo_path
        path_that_should_survive = second_embedding.photo_path
        face_store.delete_face_embedding(first_embedding.id)
        import pathlib

        _check("the removed embedding's photo file is gone", not pathlib.Path(path_to_be_removed).exists())
        _check("the OTHER embedding's photo file still exists", pathlib.Path(path_that_should_survive).exists())
        remaining = face_store.list_embeddings_for(PERSON_KEY)
        _check("two embeddings remain (one removed of three)", len(remaining) == 2, detail=str(len(remaining)))
        print()

        print("=== 5: removing the WHOLE individual removes every remaining photo file (consent-completeness) ===")
        identity_photo_dir = KNOWN_INDIVIDUAL_PHOTOS_DIR / PERSON_KEY
        _check("the identity's photo directory exists before removal", identity_photo_dir.exists())
        remaining_photo_paths = [pathlib.Path(e.photo_path) for e in remaining if e.photo_path]
        _check("there's at least one photo file left to prove gets deleted", len(remaining_photo_paths) > 0)

        face_store.delete_known_individual(PERSON_KEY)
        _check("individual is gone", face_store.get_by_identity_key(PERSON_KEY) is None)
        _check("every one of their embeddings is gone", face_store.list_embeddings_for(PERSON_KEY) == [])
        _check(
            "the ENTIRE identity photo directory is gone — no leftover photo files anywhere",
            not identity_photo_dir.exists(),
        )
        _check(
            "every individual photo file path is confirmed gone too",
            all(not p.exists() for p in remaining_photo_paths),
            detail=str([str(p) for p in remaining_photo_paths if p.exists()]),
        )
        print()

        print(f"Overall: {'ALL PASS' if ALL_PASS else 'SOME FAILED — see above'}")
    finally:
        face_store.delete_known_individual(PERSON_KEY)


if __name__ == "__main__":
    main()
