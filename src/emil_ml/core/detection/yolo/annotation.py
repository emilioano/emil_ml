"""Shared YOLO-format annotation utilities: mask-to-box conversion, label
file I/O, and box drawing for preview/review.

Used by onboarding (all three annotation paths converge on the same
images/ + labels/ pool via these functions) and by the inspect page (drawing
a trained model's detections). `YoloTrainer`/`YoloPredictor` never need to
know which path produced a given label file — by the time training runs,
it's just images + .txt files on disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

# (class_id, center_x, center_y, width, height), all normalized to [0, 1] —
# the standard YOLO label format.
YoloBox = tuple[int, float, float, float, float]


def mask_to_yolo_boxes(mask: np.ndarray, class_id: int = 0, min_area: int = 4) -> list[YoloBox]:
    """Connected-component bounding boxes from a binary/grayscale mask.

    Non-zero pixels are foreground (defect region). Each disjoint blob
    becomes its own box, so a mask with several separate defect regions
    (common in MVTec-style ground truth) produces several boxes, not one
    box spanning all of them. `min_area` (in pixels) drops specks too small
    to be a real defect — usually antialiasing/compression noise at mask
    edges rather than signal.
    """
    binary = np.asarray(mask) > 0
    if not binary.any():
        return []

    labeled, _ = ndimage.label(binary)
    height, width = binary.shape[:2]
    boxes: list[YoloBox] = []
    for region_slice in ndimage.find_objects(labeled):
        if region_slice is None:
            continue
        row_slice, col_slice = region_slice
        area = (row_slice.stop - row_slice.start) * (col_slice.stop - col_slice.start)
        if area < min_area:
            continue
        cx = (col_slice.start + col_slice.stop) / 2 / width
        cy = (row_slice.start + row_slice.stop) / 2 / height
        w = (col_slice.stop - col_slice.start) / width
        h = (row_slice.stop - row_slice.start) / height
        boxes.append((class_id, cx, cy, w, h))
    return boxes


def write_yolo_label(path: Path, boxes: list[YoloBox]) -> None:
    """Write boxes in YOLO .txt format. An empty list writes an empty file (= negative example)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cls, cx, cy, w, h in boxes]
    path.write_text("\n".join(lines), encoding="utf-8")


def read_yolo_label(path: Path) -> list[YoloBox]:
    """Read a YOLO .txt label file. Missing or empty file -> no boxes (negative example)."""
    path = Path(path)
    if not path.exists():
        return []
    boxes: list[YoloBox] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cls_str, cx_str, cy_str, w_str, h_str = line.split()[:5]
        boxes.append((int(cls_str), float(cx_str), float(cy_str), float(w_str), float(h_str)))
    return boxes


def write_classes_file(path: Path, class_names: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(class_names), encoding="utf-8")


def read_classes_file(path: Path) -> list[str]:
    path = Path(path)
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_classes_file(path: Path, new_class_names: list[str]) -> list[str]:
    """Add new class name(s) to an existing classes.txt, preserving every
    existing name's line position (= its class index).

    A YOLO label .txt file (see write_yolo_label/read_yolo_label) stores a
    bare integer class id with no name attached — the id only means
    anything by looking up that same line position in classes.txt. Every
    annotation ever saved for this component already has that lookup baked
    in, so this is deliberately the ONLY way to grow a component's class
    list after creation: new names are always appended at the end, never
    inserted, reordered, or used to replace an existing line. Duplicate
    names (case-insensitive, against either the existing list or within
    `new_class_names` itself) are rejected rather than silently
    deduplicated, since a caller asking to add a class that already exists
    is almost always a mistake worth surfacing.

    Returns the full, updated class list.
    """
    path = Path(path)
    existing = read_classes_file(path)
    existing_lower = {name.lower() for name in existing}
    cleaned: list[str] = []
    seen_lower: set[str] = set()
    for raw_name in new_class_names:
        name = raw_name.strip()
        if not name:
            raise ValueError("Class names must not be empty.")
        key = name.lower()
        if key in existing_lower or key in seen_lower:
            raise ValueError(f"Class {name!r} already exists — choose a different name.")
        seen_lower.add(key)
        cleaned.append(name)
    if not cleaned:
        raise ValueError("At least one new class name is required.")
    combined = existing + cleaned
    write_classes_file(path, combined)
    return combined


def render_boxes_on_image(
    image: Image.Image,
    boxes: list[YoloBox] | list[dict],
    class_names: list[str] | None = None,
) -> Image.Image:
    """Draw boxes on a copy of `image` for visual review.

    Accepts either raw normalized YOLO tuples (class_id, cx, cy, w, h) — from
    a label file, used for the mask-conversion preview — or the
    {"class": str, "confidence": float, "box": [x1, y1, x2, y2]} dicts
    YoloPredictor returns (already pixel xyxy) — used by the inspect page.
    One drawing implementation serves both, so the two "let the user see the
    boxes" needs (onboarding preview and inspection results) can't drift.
    """
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size

    for box in boxes:
        if isinstance(box, dict):
            x1, y1, x2, y2 = box["box"]
            label = f"{box['class']} {box['confidence']:.2f}"
        else:
            cls_id, cx, cy, w, h = box
            x1 = (cx - w / 2) * width
            y1 = (cy - h / 2) * height
            x2 = (cx + w / 2) * width
            y2 = (cy + h / 2) * height
            label = class_names[cls_id] if class_names and cls_id < len(class_names) else str(cls_id)

        draw.rectangle([x1, y1, x2, y2], outline="#FF3B30", width=3)
        draw.text((x1 + 2, max(0, y1 - 14)), label, fill="#FF3B30")

    return annotated
