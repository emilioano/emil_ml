"""Shared box-annotation canvas widget — one implementation of "show an
image, let a human draw/adjust rectangles, assign each a class, hand back
YOLO-normalized boxes" used by both app/pages/2_onboard.py's manual
annotation path (Path 3) and app/pages/3_inspection_station.py's
flag-a-prediction-as-incorrect correction flow. Not itself a Streamlit
page (lives outside app/pages/, so it's never picked up as one) — a
plain shared module both pages import.

Kept as exactly one implementation specifically so the coordinate-
conversion math (display scaling, pixel-to-normalized conversion) can
never drift between "annotate a fresh image from scratch" and "correct
this specific prediction" — the same bug fixed once benefits both.
"""

from __future__ import annotations

import streamlit as st
from PIL import Image

from emil_ml.utils.streamlit_compat import patch_image_to_url

patch_image_to_url()
from streamlit_drawable_canvas import st_canvas  # noqa: E402 (must follow the compat patch above)

from emil_ml.core.detection.yolo.annotation import YoloBox

MAX_CANVAS_DIM = 700


def render_yolo_box_canvas(
    pil_image: Image.Image,
    class_names: list[str],
    *,
    key_prefix: str,
    initial_boxes: list[YoloBox] | None = None,
) -> list[YoloBox]:
    """Render the canvas for one image and return whatever boxes are
    CURRENTLY drawn on it, as normalized YOLO tuples — live, on every
    rerun. The caller decides when to actually persist that (e.g. behind
    its own "Save" button); this function never writes anything itself.

    `initial_boxes`, if given, pre-draws those rectangles on the canvas
    so a correction starts from something instead of a blank slate —
    used by the Inspection Station when an inspection already has a
    verified_label with boxes (e.g. re-opening a correction started
    earlier in the same session). There is deliberately no way to
    pre-draw the MODEL's own predicted boxes: PredictionResult's box
    coordinates are never persisted anywhere (see
    core/inspections/orchestrator.py's run_inspection() docstring) — only
    a "what got flagged" image and class names survive that long, so
    "preloaded" here means the flagged image loads automatically (no
    re-upload step), not that a prediction's boxes are pre-drawn.
    """
    img_w, img_h = pil_image.size
    scale = min(1.0, MAX_CANVAS_DIM / max(img_w, img_h))
    disp_w, disp_h = int(img_w * scale), int(img_h * scale)
    display_image = pil_image.convert("RGB").resize((disp_w, disp_h))

    initial_drawing = None
    if initial_boxes:
        initial_drawing = {
            "version": "4.4.0",
            "objects": [
                {
                    "type": "rect",
                    "left": (cx - w / 2) * img_w * scale,
                    "top": (cy - h / 2) * img_h * scale,
                    "width": w * img_w * scale,
                    "height": h * img_h * scale,
                    "angle": 0,
                    "scaleX": 1,
                    "scaleY": 1,
                    "opacity": 1,
                    "fill": "rgba(255, 59, 48, 0.2)",
                    "stroke": "#FF3B30",
                    "strokeWidth": 2,
                }
                for _cls, cx, cy, w, h in initial_boxes
            ],
        }

    canvas_result = st_canvas(
        fill_color="rgba(255, 59, 48, 0.2)",
        stroke_width=2,
        stroke_color="#FF3B30",
        background_image=display_image,
        height=disp_h,
        width=disp_w,
        drawing_mode="rect",
        initial_drawing=initial_drawing,
        key=f"yolo_canvas_{key_prefix}",
    )

    rects_px: list[tuple[float, float, float, float]] = []
    if canvas_result.json_data is not None:
        for obj in canvas_result.json_data.get("objects", []):
            if obj.get("type") != "rect":
                continue
            # Canvas coords are in DISPLAYED (possibly downscaled) pixels —
            # divide by `scale` to recover ORIGINAL image pixel coordinates.
            # scaleX/scaleY account for the user resizing a rect after
            # drawing (or after it was pre-loaded) it.
            left = obj["left"] / scale
            top = obj["top"] / scale
            w = obj["width"] * obj.get("scaleX", 1) / scale
            h = obj["height"] * obj.get("scaleY", 1) / scale
            rects_px.append((left, top, w, h))

    if not rects_px:
        return []

    # Best-effort default class per box: if this canvas started from
    # initial_boxes, match by position (the canvas preserves draw order
    # for pre-loaded objects the user hasn't touched) so re-confirming an
    # unedited box doesn't reset its class back to index 0. Any box added
    # or reordered by the user just falls back to index 0, same as a
    # from-scratch annotation always has.
    initial_class_by_index = {i: b[0] for i, b in enumerate(initial_boxes or [])}

    st.caption(f"{len(rects_px)} box(es) drawn — assign a class to each:")
    boxes: list[YoloBox] = []
    for i, (left, top, w, h) in enumerate(rects_px):
        default_idx = initial_class_by_index.get(i, 0)
        if default_idx >= len(class_names):
            default_idx = 0
        cls_name = st.selectbox(
            f"Box {i + 1} class",
            class_names,
            index=default_idx,
            key=f"yolo_canvas_class_{key_prefix}_{i}",
        )
        class_id = class_names.index(cls_name)
        cx = (left + w / 2) / img_w
        cy = (top + h / 2) / img_h
        boxes.append((class_id, cx, cy, w / img_w, h / img_h))
    return boxes
