"""Supervised CNN classifier: trains directly on approved AND failed images.

Transfer learning on a frozen pretrained ImageNet base (see model.py), with
an optional fine-tuning phase. Learns to tell the two classes apart directly,
rather than modeling "normal" and flagging deviations.

**Good fit when:**
- You have a reasonable number of labeled examples of *both* classes —
  the more failed examples, the better it can learn what a defect actually
  looks like instead of guessing.
- The defect type(s) are known in advance and you can collect/label examples
  of them.

**Weak when:**
- Data is very limited (tens of images per class). Observed directly on real
  data: with too little signal to learn a real decision boundary, training
  can collapse toward predicting one class for almost everything — e.g. every
  real defect gets caught (recall=1.0) but so does most of the approved set
  (precision collapses), which looks like a strong recall number while being
  close to useless in practice. `ClassifierTrainer` guards against the worst
  form of this (it evaluates head-only vs. fine-tuned and keeps whichever is
  less collapsed, by balanced accuracy rather than plain recall), but it
  can't manufacture signal that isn't in the data.
- Requires failed examples at all — the autoencoder can be trained approved-
  only, this method can't.

See `trainer.py`/`model.py` for the advanced settings this method exposes
(`base_model`, `pooling`, `class_weight_strategy`, `augmentation_strength`,
`fine_tune_epochs`, `fine_tune_learning_rate`, `fine_tune_unfreeze_layers`),
configurable per-component in the onboarding UI's "Advanced settings".
`ClassifierTrainer`/`ClassifierPredictor` are registered under
("image", "classifier") in `core/registry_factory.py`.
"""
