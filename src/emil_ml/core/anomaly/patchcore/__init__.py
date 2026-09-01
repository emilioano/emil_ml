"""PatchCore: unsupervised, patch-level anomaly detection via anomalib.

Trains only on approved images — needs no defect examples at all, like the
autoencoder. Unlike the autoencoder, it never judges the image as a whole:
a frozen pretrained CNN backbone extracts a grid of local patch features,
and a memory bank of those features (built once from normal images, then
coreset-subsampled to stay small) is what "normal" is defined by. A new
image's anomaly score is how far its own patches sit from their nearest
neighbors in that memory bank.

A thin adapter around anomalib (Intel) — see `adapter.py` — not a from-scratch
implementation; anomalib provides the actual PatchCore algorithm, the MVTec-
style folder data loading, coreset subsampling, and automatic thresholding.

**Good fit when:**
- The defect is small and localized relative to the whole image — exactly
  the case the autoencoder (whole-image reconstruction error, averaged over
  every pixel) and classifier (whole-image features) both struggle with,
  since patch-level comparison doesn't dilute a small anomaly across a large
  correctly-normal background.
- You have normal examples but few or no labeled defective ones — like the
  autoencoder, any failed images you do have are used only to validate/
  calibrate the threshold, never for training.
- You want to see *where* the anomaly is, not just that there is one — see
  below.

**Weak when:**
- Normal images vary a lot on their own (lighting, pose, background clutter,
  natural texture variation) — patches from that natural variation can sit
  just as far from the memory bank as a real defect would, producing more
  false alarms than a method that's seen the whole image's context.
- Dependency weight: this is the only method in EMIL built on PyTorch via a
  third-party framework (anomalib) with its own large dependency tree
  (Lightning, kornia, timm, scikit-image, ...) rather than the TensorFlow/
  Keras stack the rest of the app uses. It's an optional install (the
  `patchcore` extra in pyproject.toml) for exactly this reason — anomalib is
  imported lazily, only when a PatchCore component is actually trained or
  inspected, so nothing else in the app requires it.

The heatmap in a prediction's `details["heatmap"]` — a per-pixel anomaly
score map, the same shape as the input image — is this method's strongest
demo property: it shows where the anomaly was found without ever having
been given a single annotated bounding box, unlike YOLO. Rendered as an
overlay on the Inspect page (see `core/anomaly/patchcore/heatmap.py`).

See `trainer.py`/`predictor.py` for the three advanced settings this method
exposes (`patchcore_backbone`, `patchcore_coreset_sampling_ratio`,
`patchcore_num_neighbors`), configurable per-component in the onboarding
UI's "Advanced settings".
"""
