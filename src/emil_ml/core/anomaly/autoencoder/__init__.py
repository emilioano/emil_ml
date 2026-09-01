"""Autoencoder: unsupervised anomaly detection via reconstruction error.

Trains only on approved images — needs no defect examples at all. Learns to
reconstruct "normal", then flags anything it reconstructs badly.

**Good fit when:**
- Defects are unpredictable or rare enough that you don't have (or can't
  enumerate) labeled failure examples.
- The anomaly is spatially large or obvious relative to the whole image —
  a big chunk of the object is wrong, missing, or a different shape/color.
- You have plenty of normal examples but few or no defective ones.

**Weak when:**
- The defect is small and localized. Reconstruction error is averaged over
  every pixel by default (`score_method="global_mean"`), so a tiny bad patch
  gets diluted by a large, correctly-reconstructed background — observed
  directly on real data (a toothbrush defect dataset) where approved and
  defective images produced nearly identical reconstruction error and only
  ~20-27% of real defects were caught. Try `score_method="local_max"` first
  if you suspect this — it scores by the single worst-reconstructed pixel
  instead of the average, which is much more sensitive to a small anomaly,
  at the cost of being noisier on images with no defect at all.

See `trainer.py`/`predictor.py` for the two advanced settings this method
exposes (`score_method`, `threshold_percentile`), configurable per-component
in the onboarding UI's "Advanced settings".
"""
