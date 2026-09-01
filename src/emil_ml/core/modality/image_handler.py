"""Image modality: decodes raw input into a PIL Image.

Deliberately does NOT resize/normalize here — that used to happen in this
handler, but different predictors want different things: the Keras-based
autoencoder/classifier want a square, [0,1]-normalized array (they do that
themselves now, in predict()), while YOLO's own preprocessing expects a
image without a forced square resize/distortion. Keeping this handler to
"decode raw input into a usable image object" and pushing final prep into
each predictor is what lets one modality handler serve every method.
"""

from __future__ import annotations

from PIL import Image

from emil_ml.core.modality.base import BaseModalityHandler
from emil_ml.utils import image_io
from emil_ml.utils.image_io import ImageInput


class ImageModalityHandler(BaseModalityHandler):
    """Loads a path/bytes/PIL/array image into a decoded RGB PIL Image."""

    def load(self, raw_input: ImageInput) -> Image.Image:
        return image_io.to_pil(raw_input).convert("RGB")
