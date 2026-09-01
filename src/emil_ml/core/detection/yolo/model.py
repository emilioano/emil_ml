"""Loads a pretrained Ultralytics YOLO model for fine-tuning/inference."""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from emil_ml.config.settings import DATA_DIR

DEFAULT_MODEL_VARIANT = "yolo11n.pt"

# Ultralytics normally downloads pretrained weights via a `curl` subprocess
# (Windows: schannel TLS). On a network with TLS interception that Windows'
# own certificate store trusts but whose revocation-check endpoint is
# unreachable (observed against github.com on this machine), that download
# fails outright — curl treats a failed revocation check as fatal. Python's
# `requests`, patched via `truststore` to verify against the OS certificate
# store instead of a bundled CA list, gets past the same interception fine.
# So: pre-fetch into our own deterministic cache dir via requests+truststore,
# then hand YOLO() an absolute path to the already-downloaded file — it uses
# a given file as-is and never attempts its own download in that case.
_WEIGHTS_DIR = DATA_DIR / "yolo_weights"
_ASSETS_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0"


def _ensure_weights_downloaded(model_variant: str) -> Path:
    """Return a local path to `model_variant`'s weights, downloading them if needed.

    Falls back to returning the bare filename (letting YOLO() attempt its
    own download) if our pre-fetch fails for any reason — no worse off than
    without this workaround.
    """
    _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = _WEIGHTS_DIR / model_variant
    if dest.exists():
        return dest

    try:
        import truststore

        truststore.inject_into_ssl()
        import requests

        response = requests.get(f"{_ASSETS_URL}/{model_variant}", stream=True, timeout=60)
        response.raise_for_status()
        tmp_path = dest.with_suffix(dest.suffix + ".part")
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        tmp_path.rename(dest)
    except Exception:
        return Path(model_variant)
    return dest


def load_yolo_model(model_variant: str = DEFAULT_MODEL_VARIANT) -> YOLO:
    """Load a pretrained YOLO checkpoint, pre-fetching it ourselves if not already cached."""
    weights_path = _ensure_weights_downloaded(model_variant)
    return YOLO(str(weights_path))
