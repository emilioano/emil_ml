"""Shared building blocks for BaseTrainer.evaluate() (core/base.py) —
model_type-aware evaluation reporting, generated and saved after every
successful training run (see training/onboard.py's train_component(),
gated by the component's generate_evaluation_report setting).

This package holds only what's genuinely shared across methods:

- plots.py — matplotlib helpers for the chart *kinds* multiple model_types
  need (confusion matrix, a named-series curve over epochs, a score
  histogram, ROC/PR curves) — model_type-agnostic; each trainer's own
  evaluate() decides which of these to call and with what data.
- io.py — writing metrics.json/CSV, the machine-readable half of a report.
- unsupervised.py — the one genuinely shared *evaluation procedure*: given
  approved/failed anomaly scores and a threshold, produce the histogram +
  (if labeled failures exist) ROC/PR curves + threshold precision/recall/
  F1. core/anomaly/autoencoder, core/anomaly/patchcore, and
  core/anomaly/isolation_forest's evaluate() methods all call this — it's
  the same procedure for all three, only how each method PRODUCES its raw
  scores differs (reconstruction error, PatchCore's normalized score,
  Isolation Forest's negated score_samples). Loss-curve plotting is
  deliberately NOT part of this shared procedure — only the autoencoder
  trains iteratively; PatchCore and Isolation Forest do not, and must not
  get a loss curve (see autoencoder/trainer.py's own evaluate()).

Nothing here knows about any specific model_type. A trainer's evaluate()
imports what it needs and decides its own artifact set — see
core/base.py's BaseTrainer.evaluate() docstring for why that's a
deliberate, per-method decision rather than a fixed template.
"""
