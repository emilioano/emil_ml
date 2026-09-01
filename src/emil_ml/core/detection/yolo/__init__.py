"""YOLO object detection: fine-tunes a pretrained Ultralytics model to
localize specific objects/defects, rather than judging a whole image.

Needs annotated data (images + bounding boxes), which onboarding can produce
three ways — see `training/onboard.py`'s YOLO section:

1. Already in YOLO format: images + ready-made .txt labels, used as-is.
2. Mask-to-box conversion: upload images + segmentation masks (MVTec-style);
   `annotation.mask_to_yolo_boxes()` runs connected-component extraction to
   get bounding boxes, with a visual preview before it's saved.
3. Manual annotation: draw boxes by hand on raw images in the app
   (`streamlit-drawable-canvas`), assigning a class per box.

All three write into the same images/+labels/ pool (`utils/paths.py`
`yolo_images_dir`/`yolo_labels_dir`/`yolo_classes_file`) via
`training.onboard.add_yolo_annotation()`. `YoloTrainer`/`YoloPredictor` never
see which path produced a given label — by training time it's just files on
disk, identical regardless of origin.

`YoloTrainer`/`YoloPredictor` are registered under ("image", "yolo") in
`core/registry_factory.py`.
"""
