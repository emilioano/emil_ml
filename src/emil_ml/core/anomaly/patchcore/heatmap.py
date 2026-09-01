"""Renders a PatchCore anomaly heatmap as a color overlay on the input image.

Deliberately has no anomalib import — pure PIL/numpy — so the Inspect page
can render a PatchCore result's heatmap without needing anomalib installed
at all (a `PredictionResult.details["heatmap"]` is already a plain numpy
array by the time it reaches this module; see adapter.py/predictor.py).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Blue (low anomaly score) -> red (high), a small hand-rolled colormap so
# this module doesn't need matplotlib as a dependency just for this.
_COLORMAP_STOPS = (
    (0.0, (0, 0, 160)),
    (0.35, (0, 180, 255)),
    (0.6, (255, 255, 0)),
    (1.0, (220, 0, 0)),
)


def _colorize(values: np.ndarray) -> np.ndarray:
    """Map a (H, W) float array in [0, 1] to an (H, W, 3) uint8 RGB array."""
    values = np.clip(values, 0.0, 1.0)
    rgb = np.zeros((*values.shape, 3), dtype=np.float32)
    for (stop_a, color_a), (stop_b, color_b) in zip(_COLORMAP_STOPS, _COLORMAP_STOPS[1:]):
        span = stop_b - stop_a
        t = np.clip((values - stop_a) / span, 0.0, 1.0)
        in_segment = (values >= stop_a) & (values <= stop_b)
        for channel in range(3):
            segment_color = color_a[channel] + t * (color_b[channel] - color_a[channel])
            rgb[..., channel] = np.where(in_segment, segment_color, rgb[..., channel])
    return rgb.astype(np.uint8)


def render_heatmap_overlay(image: Image.Image, heatmap: np.ndarray, *, alpha: float = 0.45) -> Image.Image:
    """Alpha-blend a colorized `heatmap` (any (H, W), values roughly 0-1) onto `image`.

    Resizes the heatmap to `image`'s size first (they're rarely the same
    resolution — PatchCore's internal pre_processor works at its own fixed
    size regardless of the input image's native resolution).
    """
    image = image.convert("RGB")
    heatmap_img = Image.fromarray(_colorize(heatmap), mode="RGB").resize(image.size, Image.BILINEAR)
    return Image.blend(image, heatmap_img, alpha=alpha)
