"""Image loading, preprocessing, and saving utilities."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

ImageInput = Union[str, Path, bytes, bytearray, Image.Image, np.ndarray]

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def to_pil(image: ImageInput) -> Image.Image:
    """Coerce a path, raw bytes, PIL image, or numpy array into a PIL Image."""
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (str, Path)):
        return Image.open(image)
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image))
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    raise TypeError(f"Unsupported image input type: {type(image)!r}")


def normalize(pil: Image.Image, size: int) -> np.ndarray:
    """Resize a PIL image to a square and normalize it to a float32 (size, size, 3) array in [0, 1].

    Split out from `preprocess()` so callers that already have a PIL image
    (e.g. a modality handler's output) don't have to round-trip through
    `to_pil()` again, and so a predictor that doesn't want this square/
    normalized form (e.g. YOLO, which does its own preprocessing internally
    and works better on a non-distorted image) can skip it entirely.
    """
    resized = pil.convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def preprocess(image: ImageInput, size: int) -> np.ndarray:
    """Load, resize, and normalize an image to a float32 (size, size, 3) array in [0, 1]."""
    return normalize(to_pil(image), size)


def load_dataset(directory: Path, size: int) -> np.ndarray:
    """Load every image file in `directory` into a (N, size, size, 3) float32 array."""
    directory = Path(directory)
    paths = sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )
    if not paths:
        return np.empty((0, size, size, 3), dtype=np.float32)
    return np.stack([preprocess(p, size) for p in paths])


def save_image(image: ImageInput, path: Path) -> None:
    """Save an image (in any supported input form) to `path`."""
    to_pil(image).save(path)


def save_upload(data: bytes, dest_dir: Path, filename: str) -> Path:
    """Write raw uploaded bytes to `dest_dir/filename` and return the path."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename
    dest_path.write_bytes(data)
    return dest_path
