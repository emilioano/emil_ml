"""Object detection (e.g. YOLO) for localizing specific objects/defects within images.

Unlike the autoencoder (whole-image anomaly score) or the classifier
(whole-image approved/failed), detection draws a bounding box around the
specific thing it found, rather than judging the image as a whole.

**Good fit when:**
- The defect (or expected object) is small and localized — a foreign object,
  a specific visible flaw — and the rest of the image is identical between
  classes. This is exactly the failure mode where the autoencoder and
  classifier both struggle (a small anomaly gets diluted by everything
  around it); localizing to a region instead of scoring the whole image
  sidesteps that.
- You can tell (and annotate) *where* the thing of interest is, not just
  whether the image is good or bad.

**Weak when:**
- You don't have bounding-box-annotated training data — this needs more
  upfront preparation than the other two methods (which only need images
  sorted into approved/failed folders).
- The anomaly isn't a discrete localizable object/region (e.g. a diffuse
  color shift across the whole image) — a classifier is a better fit there.

See `core/detection/yolo/` for the implementation (`YoloTrainer`/
`YoloPredictor`, registered under ("image", "yolo") in
`core/registry_factory.py`).
"""
