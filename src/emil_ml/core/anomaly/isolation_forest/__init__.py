"""Isolation Forest: unsupervised anomaly detection on CNN embeddings.

Trains only on approved images — needs no defect examples at all, like the
autoencoder and PatchCore. Rather than working on pixels directly, it reuses
the same frozen ImageNet-pretrained CNN backbone the UMAP/PCA diagnostics
page uses (`core/diagnostics/embeddings.py`) to reduce each whole image to a
single feature vector, then fits an sklearn `IsolationForest` on the
approved set's vectors. A new image is anomalous if its embedding is
"easy to isolate" relative to that fitted forest — few random partitions
needed to separate it from the rest, in the algorithm's own terms.

**Good fit when:**
- The defect changes the image's overall character (missing part, wrong
  color/texture, wrong orientation) enough to shift its whole-image
  embedding — the same class of anomaly the autoencoder is good at.
- You want something simple and fast to train/iterate on: no epochs, no
  GPU-bound training loop, just fitting a forest on a small number of
  feature vectors (typically seconds, not minutes).
- You've already used the diagnostics page to confirm approved/failed
  embeddings separate reasonably well in UMAP/PCA space — that's directly
  the same embedding space this method scores in.

**Weak when:**
- The defect is small and localized relative to the whole image — like the
  autoencoder and classifier, averaging/pooling the whole image into one
  feature vector can wash out a small local anomaly that doesn't meaningfully
  shift the overall embedding. PatchCore's patch-level comparison is the
  better fit for that case.

See `trainer.py`/`predictor.py` for the advanced settings this method
exposes (`isolation_forest_n_estimators`, `isolation_forest_contamination`,
`isolation_forest_max_features`, `isolation_forest_standardize`),
configurable per-component in the onboarding UI's "Advanced settings".
"""
