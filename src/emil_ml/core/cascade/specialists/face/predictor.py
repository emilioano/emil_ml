"""FaceRecognitionSpecialist: detect the most prominent face within a
region, embed it, and match it against the known-individuals database
(store.py).

Scope note: identifies the single most prominent face per call (MTCNN's
default `keep_all=False` — highest-probability detection), matching the
cascade's one-SpecialistResult-per-detected-object shape (see
core/cascade/base.py) — core/cascade/pipeline.py calls identify() once
per detected "human" object (see its own module docstring), so multiple
people in one frame already means multiple identify() calls, each scoped
to its own person's box; this specialist itself still only ever looks for
one face per call.

Cropping to `box` (see identify()) is what makes that per-person scoping
real: since COCO-YOLO replaced the old whole-frame ImageNet classifier as
the cascade's coarse stage (see core/detection/yolo_coco), each "human"
detection carries its own bounding box, so this specialist can search
within just that person's region instead of the whole frame — sharper
when there are several people in one image, and cheaper too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from emil_ml.config.settings import DEFAULT_FACE_MATCH_DISTANCE_THRESHOLD
from emil_ml.core.cascade.base import BaseSpecialist, BoundingBox, SpecialistResult
from emil_ml.core.cascade.specialists.face import store

# Expand a detected object's box by this fraction (each side) before
# cropping — a "person" box from a detector is usually tight around the
# whole body, and padding guards against it clipping the top of the head/
# hairline right where a face crop needs the most margin. Bounded to the
# source image's own extents (see _crop_with_padding), so this never
# reads outside the actual frame.
_BOX_PADDING_FRACTION = 0.15

_MTCNN: Any = None
_RESNET: Any = None
_MTCNN_ALL_FACES: Any = None  # separate instance: keep_all=True, only for detect_all_faces() below


def _load_models() -> tuple[Any, Any]:
    global _MTCNN, _RESNET
    if _MTCNN is None or _RESNET is None:
        from facenet_pytorch import MTCNN, InceptionResnetV1

        _MTCNN = MTCNN(keep_all=False)
        _RESNET = InceptionResnetV1(pretrained="vggface2").eval()
    return _MTCNN, _RESNET


@dataclass(frozen=True)
class DetectedFace:
    """One face MTCNN found in an image, with its own embedding — the
    building block for the Onboard page's known-individuals registration
    UI, which needs to show EVERY detected face (not just the single most
    prominent one identify() cares about) so an operator can confirm the
    right one before registering it — e.g. picking the correct person out
    of a group photo, not the background or someone else in frame."""

    box: BoundingBox
    confidence: float
    embedding: list[float]


def detect_all_faces(image: Any) -> list[DetectedFace]:
    """Detect and embed every face in `image` — used only by the
    registration UI (see app/pages/2_onboard.py), never by the cascade
    itself (identify() above stays single-face, matching
    core/cascade/pipeline.py's one-call-per-detected-object shape).
    Returns [] if no face was found; never raises for that case, same
    "no match is still a result" convention as identify().

    Uses a SEPARATE cached MTCNN instance (keep_all=True) from
    identify()'s own (keep_all=False, fixed at construction) — sharing
    the same cached InceptionResnetV1 for the actual embedding step,
    since that part doesn't depend on how many faces were detected.
    """
    global _MTCNN_ALL_FACES
    import torch

    _, resnet = _load_models()
    if _MTCNN_ALL_FACES is None:
        from facenet_pytorch import MTCNN

        _MTCNN_ALL_FACES = MTCNN(keep_all=True)

    boxes, probs = _MTCNN_ALL_FACES.detect(image)
    if boxes is None:
        return []

    # facenet-pytorch's own __call__ (not .detect()) returns the aligned,
    # cropped face tensors ready for the embedding model — confirmed
    # directly that its face order matches .detect()'s box order, so
    # zipping the two by index is safe (not an assumption).
    face_tensors = _MTCNN_ALL_FACES(image)
    if face_tensors is None:
        return []

    with torch.no_grad():
        embeddings = resnet(face_tensors)

    return [
        DetectedFace(box=tuple(float(v) for v in box), confidence=float(prob), embedding=embedding.tolist())
        for box, prob, embedding in zip(boxes, probs, embeddings)
    ]


class FaceRecognitionSpecialist(BaseSpecialist):
    name = "face"

    def __init__(self, *, match_threshold: float = DEFAULT_FACE_MATCH_DISTANCE_THRESHOLD) -> None:
        self._match_threshold = match_threshold

    def identify(self, image: Any, *, box: BoundingBox | None = None) -> SpecialistResult:
        import torch

        mtcnn, resnet = _load_models()

        search_image = _crop_with_padding(image, box) if box is not None else image
        face_tensor = mtcnn(search_image)
        if face_tensor is None:
            return SpecialistResult(
                matched=False,
                identity_key="unknown",
                identity_label="Unknown",
                confidence=None,
                details={"reason": "no_face_detected", "searched_box": box},
            )

        with torch.no_grad():
            embedding = resnet(face_tensor.unsqueeze(0))[0].tolist()

        match = store.find_best_match(embedding, threshold=self._match_threshold)
        if match is None:
            return SpecialistResult(
                matched=False,
                identity_key="unknown",
                identity_label="Unknown",
                confidence=None,
                details={"reason": "no_match_within_threshold", "threshold": self._match_threshold, "searched_box": box},
            )

        known_individual, distance = match
        return SpecialistResult(
            matched=True,
            identity_key=known_individual.identity_key,
            identity_label=known_individual.name,
            confidence=max(0.0, 1.0 - distance / self._match_threshold),
            details={"distance": distance, "threshold": self._match_threshold, "searched_box": box},
        )


def _crop_with_padding(image: Image.Image, box: BoundingBox, *, pad_fraction: float = _BOX_PADDING_FRACTION) -> Image.Image:
    """Crop `image` to `box`, expanded by `pad_fraction` on each side and
    clamped to the image's own extents — see module-level
    _BOX_PADDING_FRACTION for why the padding exists."""
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    pad_x, pad_y = width * pad_fraction, height * pad_fraction
    left = max(0.0, x1 - pad_x)
    top = max(0.0, y1 - pad_y)
    right = min(float(image.width), x2 + pad_x)
    bottom = min(float(image.height), y2 + pad_y)
    return image.crop((left, top, right, bottom))
