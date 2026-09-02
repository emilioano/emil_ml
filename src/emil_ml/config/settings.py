"""Global paths and app-level defaults for EMIL."""

from __future__ import annotations

import os
from pathlib import Path

# Project root = two levels up from this file (src/emil_ml/config/settings.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
COMPONENTS_DIR = DATA_DIR / "components"
DB_PATH = PROJECT_ROOT / "emil.db"

# Default per-component model settings (mirrored in the `components` table schema)
DEFAULT_IMAGE_SIZE = 256
DEFAULT_EPOCHS = 50
DEFAULT_LATENT_DIM = 128
DEFAULT_BATCH_SIZE = 32

# Shared across every trainable method (autoencoder + both classifier phases):
# stop a training run once its monitored loss hasn't improved for this many
# consecutive epochs, restoring the best epoch's weights rather than
# whatever the last epoch happened to produce. Guards against overfitting an
# already-small dataset by training past the point it's actually helping.
# 0 disables it and always runs the full epoch budget.
DEFAULT_EARLY_STOPPING_PATIENCE = 8

# Pluggable analysis methods — all seven implemented.
DEFAULT_MODEL_TYPE = "autoencoder"
VALID_MODEL_TYPES = (
    "autoencoder", "isolation_forest", "classifier", "yolo", "patchcore", "resnet_classifier", "coco_detector",
)

# --- Autoencoder advanced settings (core/anomaly/autoencoder) -----------------
# How the reconstruction-error score is computed from the per-pixel error map.
# "global_mean": average error over the whole image — the original approach.
# Works when an anomaly changes a large fraction of the image. "local_max":
# the single worst-reconstructed pixel — better for a small, localized defect,
# since averaging over the whole (correctly-reconstructed) background dilutes
# it to near-nothing (observed on real data: good/defective barely separated
# under global_mean because the defect was small relative to the frame).
DEFAULT_SCORE_METHOD = "global_mean"
VALID_SCORE_METHODS = ("global_mean", "local_max")

# Percentile of the approved reconstruction-error distribution used as the
# anomaly threshold. Higher = stricter default threshold (fewer approved
# images flagged, but also fewer real defects caught); lower = more sensitive.
DEFAULT_THRESHOLD_PERCENTILE = 97.5

# Input modality, orthogonal to model_type. Only "image" is implemented; a
# "text" handler is stubbed in core/modality/text_handler.py for a later
# log/text-analysis phase. registry_factory dispatches on (modality, model_type).
DEFAULT_MODALITY = "image"
VALID_MODALITIES = ("image", "text")

# --- Classifier advanced settings (core/classification/cnn_classifier) --------
# Not used by other model_types. Validated by ClassifierTrainer/build_classifier
# themselves, not here.

# Pretrained backbone used for transfer learning.
DEFAULT_CLASSIFIER_BASE_MODEL = "mobilenet_v2"
VALID_CLASSIFIER_BASE_MODELS = ("mobilenet_v2", "efficientnet_b0")

# How the base model's feature map is pooled before the dense head. "max"
# keeps the strongest local activation regardless of how little of the image
# it covers (better for small, localized defects); "average" smooths over the
# whole feature map. Empirically "average" has outperformed "max" on the one
# real dataset tested so far — "max" needs more data to learn reliably which
# activations are real signal vs. noise, since it only backprops through a
# single spatial location per step.
DEFAULT_CLASSIFIER_POOLING = "average"
VALID_CLASSIFIER_POOLING_TYPES = ("max", "average")

# How training-sample loss is weighted to counter class imbalance (e.g. more
# approved than failed examples). "balanced": weight each class inversely to
# its frequency (n_total / (n_classes * n_class)) — the default, prevents the
# majority class from dominating the loss. "none": equal weight — try this if
# "balanced" is overcorrecting (e.g. collapsing to "always predict the
# minority class").
DEFAULT_CLASS_WEIGHT_STRATEGY = "balanced"
VALID_CLASS_WEIGHT_STRATEGIES = ("balanced", "none")

# Strength (0-1) of the random rotation/zoom/brightness augmentation applied
# during training (as a fraction of full range, e.g. 0.06 ~= +/-22 degrees of
# rotation). Kept mild by default: for a subtle, localized defect, aggressive
# augmentation can distort or crop away the one cue that distinguishes the
# classes more often than it helps the model generalize. Raise it if
# overfitting; lower it (even to 0) if a defect might be getting destroyed by
# augmentation before the model ever sees it clearly.
DEFAULT_AUGMENTATION_STRENGTH = 0.06

# Fine-tuning phase (after head-only training): unfreeze the base model's top
# N layers and continue training at a low learning rate, so features adapt to
# this component's images instead of staying generic ImageNet features. Not
# guaranteed to help on tiny datasets — the trainer evaluates both the
# head-only and fine-tuned model and keeps whichever validates better, so
# raising these is safe to experiment with.
DEFAULT_FINE_TUNE_EPOCHS = 25
DEFAULT_FINE_TUNE_LEARNING_RATE = 3e-5
DEFAULT_FINE_TUNE_UNFREEZE_LAYERS = 30

# --- YOLO advanced settings (core/detection/yolo) ------------------------------
# Not used by other model_types. Validated by YoloTrainer/YoloPredictor
# themselves, not here.

# Pretrained Ultralytics checkpoint fine-tuned from. "n" (nano) is the
# smallest/fastest to fine-tune — a sensible default for the small annotated
# datasets this is meant for; "s" (small) trades speed for a bit more
# capacity if nano underfits.
DEFAULT_YOLO_MODEL_VARIANT = "yolo11n.pt"
VALID_YOLO_MODEL_VARIANTS = ("yolo11n.pt", "yolo11s.pt")

# What a detection (of any trained class, above the confidence threshold —
# reuses the generic `anomaly_threshold` column, like the classifier's
# decision threshold) means for the verdict. "presence": the model looks for
# something that shouldn't be there — found = failed (e.g. a defect, a
# foreign object). "absence": the model looks for something that should be
# there — found = approved, not found = failed (e.g. a required part).
DEFAULT_YOLO_DECISION_RULE = "presence"
VALID_YOLO_DECISION_RULES = ("presence", "absence")

# Probability [0, 1] that Ultralytics' mosaic augmentation (splicing 4 training
# images into one composite) is applied to a given batch. Ultralytics defaults
# this to 1.0 (tuned for COCO-scale datasets), but with the small, few-dozen-
# image datasets this app is built for, mosaic can slice a already-scarce
# small/localized defect region apart across the composite instead of helping
# the model generalize — the same concern that kept the classifier's
# augmentation_strength conservative. Default to 0 (disabled); raise it if a
# larger annotated dataset later warrants it.
DEFAULT_YOLO_MOSAIC = 0.0

# Ultralytics' `cls` loss-gain hyperparameter: how strongly training penalizes
# classification mistakes relative to box-position mistakes (`box` gain,
# 7.5) and box-shape mistakes (`dfl` gain, 1.5) — both left at Ultralytics'
# own defaults since only this one was asked for. Named "class loss weight"
# rather than reusing the classifier's "class_weight_strategy" terminology
# on purpose: with a single "defect" class this rebalances
# classification-vs-localization priority in the loss, not per-class
# inverse-frequency weighting the way the classifier's setting does — it's
# a different knob despite the similar name.
DEFAULT_YOLO_CLASS_LOSS_WEIGHT = 0.5  # Ultralytics' own default for `cls`

# Strength (0-1) of YOLO's geometric + color-jitter augmentation (rotation,
# translation, scaling, shear, hue/saturation/value), scaled together from a
# moderate baseline profile at strength=1.0 — same idea as the classifier's
# augmentation_strength. Horizontal flip is left at Ultralytics' own default
# regardless of this setting, same reasoning as the classifier: it can't
# destroy a defect's visibility, only mirror its position. Defaults to 0
# (disabled), same rationale as DEFAULT_YOLO_MOSAIC — a small/localized
# defect can be jittered away entirely on the few-dozen-image datasets this
# app targets, worse than getting no augmentation at all.
DEFAULT_YOLO_AUGMENTATION_STRENGTH = 0.0

# Which optimizer Ultralytics trains with. "auto" (Ultralytics' own default,
# used here previously since the trainer never passed `optimizer` at all)
# picks both the optimizer *and* its learning rate/momentum itself based on
# the model and dataset size — and in doing so, deliberately ignores
# `yolo_learning_rate` below (Ultralytics logs this explicitly: "'optimizer=
# auto' found, ignoring 'lr0=...' ... determining best 'optimizer', 'lr0'
# and 'momentum' automatically"). Pick a fixed optimizer (e.g. "SGD" or
# "AdamW") if you actually want yolo_learning_rate to take effect.
DEFAULT_YOLO_OPTIMIZER = "auto"
VALID_YOLO_OPTIMIZERS = ("auto", "SGD", "Adam", "AdamW", "Adamax", "NAdam", "RAdam", "RMSProp")

# Initial learning rate (Ultralytics' `lr0`). Only takes effect if
# yolo_optimizer is set to something other than "auto" — see above.
# Ultralytics' own default.
DEFAULT_YOLO_LEARNING_RATE = 0.01

# --- PatchCore advanced settings (core/anomaly/patchcore) ----------------------
# Not used by other model_types. Validated by PatchCoreTrainer/adapter
# themselves, not here. anomalib is an optional dependency (see the
# `patchcore` extra in pyproject.toml) — these are just plain values, so
# importing this module never requires anomalib to be installed.

# Pretrained CNN backbone features are extracted from (frozen, not
# fine-tuned — PatchCore never trains the backbone, only builds a memory
# bank from its features). wide_resnet50_2 is anomalib's own default and is
# the most thoroughly benchmarked on MVTec-style data; resnet18 trades some
# accuracy for a much smaller/faster memory bank and feature extraction,
# worth trying first on a slow machine or a very small dataset.
DEFAULT_PATCHCORE_BACKBONE = "wide_resnet50_2"
VALID_PATCHCORE_BACKBONES = ("wide_resnet50_2", "resnet18")

# Fraction of extracted normal-image patch features kept in the memory bank
# after coreset subsampling. Lower = smaller/faster memory bank and inference,
# at some risk of losing rare-but-normal patch patterns (more false alarms on
# unusual-but-fine regions); higher = more faithful memory bank, slower
# nearest-neighbor search at inference. anomalib's own default.
DEFAULT_PATCHCORE_CORESET_SAMPLING_RATIO = 0.1

# How many nearest neighbors in the memory bank a test patch is compared
# against when computing its anomaly score. anomalib's own default.
DEFAULT_PATCHCORE_NUM_NEIGHBORS = 9

# --- Isolation Forest advanced settings (core/anomaly/isolation_forest) --------
# Not used by other model_types. Validated by IsolationForestTrainer itself,
# not here. Trains on CNN embeddings (via core/diagnostics/embeddings.py,
# the same frozen-backbone extractor the diagnostics page uses), not raw
# pixels — so unlike the autoencoder there's no image_size/latent_dim
# equivalent to expose here; the embedding backbone's own native input size
# is fixed (see embeddings.DEFAULT_IMAGE_SIZE).

# Number of trees. sklearn's own default; more trees = more stable scores at
# the cost of training/inference time — rarely the first thing worth tuning.
DEFAULT_ISOLATION_FOREST_N_ESTIMATORS = 100

# Expected fraction of training data that's actually anomalous — directly
# controls where the automatic decision threshold lands (see trainer.py:
# threshold = -model.offset_, and offset_ is contamination-driven). "auto"
# uses scikit-learn's fixed heuristic from the original Isolation Forest
# paper rather than a specific fraction; pick a small numeric fraction
# instead if you have a rough sense of how rare real defects are and want
# the threshold to reflect that directly. Kept to a fixed preset list in the
# UI (not a free-value slider) since "auto" and a numeric fraction are two
# different modes, not points on the same continuous scale.
DEFAULT_ISOLATION_FOREST_CONTAMINATION = "auto"
VALID_ISOLATION_FOREST_CONTAMINATION_OPTIONS = ("auto", "0.01", "0.05", "0.1", "0.15", "0.2", "0.25")

# Fraction of embedding dimensions each tree is trained on. sklearn's own
# default (1.0 = every tree sees all features) — lower values add
# randomness/diversity between trees, which mainly matters with very
# high-dimensional embeddings.
DEFAULT_ISOLATION_FOREST_MAX_FEATURES = 1.0

# Whether to standardize embeddings (zero mean, unit variance per dimension)
# before fitting/scoring — Isolation Forest's random splits are sensitive to
# the raw scale of each feature, so without this, dimensions with
# naturally larger magnitude can dominate splitting decisions regardless of
# whether they're actually more informative. On by default; the fitted
# scaler is saved alongside the model so inference applies the identical
# transform (see trainer.py/predictor.py) — never fit independently at
# inference time, which would silently drift from what the model was
# trained on.
DEFAULT_ISOLATION_FOREST_STANDARDIZE = True

# --- ResNet-50 coarse classifier (core/classification/resnet_coarse) -----------
# Frozen, off-the-shelf ImageNet-1k weights — this model_type never trains;
# see resnet_coarse/trainer.py, whose train() is a fast no-op that only
# flips a component's status to 'ready'. It has no anomaly_threshold/
# model_path of its own — it's a classifier, not an anomaly detector.
#
# NO LONGER WIRED AS THE CASCADE'S STEP 1 (see core/cascade) — superseded
# by model_type='coco_detector' below, for a concrete, verified reason:
# ImageNet-1k's 1000 classes are overwhelmingly objects/animals, not
# generic "person" — there is no class simply called "person". A
# whole-image ResNet-50 classifier is therefore a genuinely weak signal
# for "a human is in this frame" (verified empirically: a real photo of a
# person tops out around bobsled/ski/go-kart at ~25% confidence —
# plausible context objects, not the person), which made the cascade's
# person -> face-recognition branch permanently unreachable in practice.
# This model_type is NOT removed and remains fully valid/usable on its
# own (still a real classifier, still registered in registry_factory) —
# just no longer the cascade's coarse stage. See
# core/detection/yolo_coco/__init__.py for the replacement's own
# reasoning, and that same module's docstring for how this classifier
# could still serve as an optional, secondary fine-grained classifier
# behind it later (COCO says "dog"; ImageNet can sometimes name the
# breed) — not built, just a documented option.
#
# Below this top-1 confidence, the predicted ImageNet class is reported as
# the "uncertain" coarse category instead of being forced into a bucket the
# model isn't actually confident about (no specialist is ever registered
# for "uncertain" — see core/cascade/specialist_registry.py — so this is a
# normal, silent stop, not an error).
DEFAULT_RESNET_CONFIDENCE_THRESHOLD = 0.15

# --- COCO-YOLO coarse detector (core/detection/yolo_coco) -----------------------
# Frozen, off-the-shelf COCO-pretrained YOLO weights (same machinery as
# model_type='yolo', the component-specific defect detector — just a
# stock checkpoint, never fine-tuned; see yolo_coco/trainer.py's no-op
# train(), same reasoning as the ResNet classifier above). THIS is the
# cascade framework's Step 1 now (core/cascade) — "what's broadly in this
# frame, and where" — chosen specifically because COCO's 80 classes
# include "person" directly (one of its best-represented classes), unlike
# ImageNet-1k above, plus everyday animals and vehicles, all with real
# bounding boxes rather than a whole-frame guess. Reuses
# core/detection/yolo/model.py's weight-fetching (yolo_model_variant,
# below) — the exact same pretrained-checkpoint download this project
# already relies on for the defect detector before it gets fine-tuned.
#
# Below this per-detection confidence, a candidate box is dropped entirely
# rather than kept and mapped to a category — unlike the ResNet classifier
# (one whole-frame guess, so "not confident" becomes its own "uncertain"
# category), a low-confidence YOLO detection is more usefully treated as
# "probably not a real object" and simply excluded, the same way
# model_type='yolo' already treats its own confidence floor. A frame with
# zero surviving detections reports CATEGORY_UNCERTAIN as its summary
# `verdict` (see yolo_coco/predictor.py) — the cascade itself only ever
# acts on `details["detections"]`, which is an empty list in that case.
DEFAULT_COCO_CONFIDENCE_THRESHOLD = 0.4

# Subfolders created under each component directory
TRAINING_SUBDIR = "training"
APPROVED_SUBDIR = "approved"
FAILED_SUBDIR = "failed"
MODELS_SUBDIR = "models"
INPUT_SUBDIR = "input"
ANALYZED_SUBDIR = "analyzed"
# Per-training-run evaluation artifacts (metrics.json + plots) — see
# utils/paths.py's ComponentPaths.evaluation_run_dir() and core/base.py's
# BaseTrainer.evaluate(). Not pre-created for every component (unlike the
# subdirs above); see evaluation_run_dir()'s own docstring.
EVALUATION_SUBDIR = "evaluation"
# Files the watcher (emil_ml/watcher/) couldn't process — corrupt/unreadable
# input, a not-ready component, or any other unexpected error. Sits
# alongside analyzed/approved/analyzed/failed, not inside either: an error
# is neither a verdict nor a successfully persisted inspection, and the
# file must never silently disappear or get retried forever from input/.
ERROR_SUBDIR = "error"
# Raw .md/.txt knowledge-base documents for this component (manuals, spec
# sheets, incident reports, machine-context notes) — see
# core/reporting/knowledge/indexer.py. Documentation about this component,
# not training data; kept alongside it rather than in a central directory
# so it lives and moves with the component itself.
KNOWLEDGE_SUBDIR = "knowledge"
# Final resting place for inspections after an operator acknowledges them
# — see core/inspections/lifecycle.py. Date-partitioned underneath this
# (archive/<year>/<month>/<day>/), not a fixed set of subdirectories, so
# no constant for those parts.
ARCHIVE_SUBDIR = "archive"

# Per-method model artifact filenames within a component's models/ directory
# (see utils/paths.py ComponentPaths.model_file). Kept distinct per method so
# e.g. a classifier's .keras file isn't confusingly named "autoencoder.keras",
# and so a future method needing a different format (e.g. YOLO's .pt) just
# adds an entry here.
AUTOENCODER_MODEL_FILENAME = "autoencoder.keras"
CLASSIFIER_MODEL_FILENAME = "classifier.keras"
YOLO_MODEL_FILENAME = "best.pt"
# A Lightning checkpoint (contains the frozen backbone weights, the coreset
# memory bank, and the calibrated threshold/normalization stats together) —
# see core/anomaly/patchcore/adapter.py.
PATCHCORE_MODEL_FILENAME = "patchcore.ckpt"

# sklearn objects (joblib, not .keras — these aren't Keras models).
# ISOLATION_FOREST_SCALER_FILENAME only exists on disk when
# isolation_forest_standardize was on at training time.
ISOLATION_FOREST_MODEL_FILENAME = "isolation_forest.joblib"
ISOLATION_FOREST_SCALER_FILENAME = "isolation_forest_scaler.joblib"

# Subfolder (under training/) holding a YOLO component's annotation pool —
# see utils/paths.py for the images/labels/classes layout. Rebuilt into a
# train/val split fresh each training run; this pool is the persistent
# source of truth regardless of which of the three annotation paths
# (pre-made YOLO labels, mask conversion, manual drawing) produced it.
YOLO_ANNOTATIONS_SUBDIR = "yolo"
YOLO_DATASET_SUBDIR = "yolo_dataset"

VALID_STATUSES = ("created", "training", "ready", "failed")

# --- RAG report generation (core/reporting) --------------------------------
# A post-detection step, not a model_type: takes a PredictionResult, adds
# retrieved documentation + machine-context anomalies, and produces a
# ReportResult. Method-agnostic — works the same regardless of which
# model_type produced the PredictionResult it's given.

# Source markdown/text documents live under each component's own
# knowledge/ subdirectory (data/components/<name>/knowledge/*.md — see
# utils/paths.py's ComponentPaths.knowledge_dir and
# core/reporting/knowledge/indexer.py), not a central directory —
# documentation for a component travels with it, same as its training
# data and models. index_all() gets the list of components to index from
# the component registry, not by scanning a directory.

# Where ChromaDB persists its index (separate from the source documents
# above — this is derived/rebuildable, not source of truth).
CHROMA_PERSIST_DIR = DATA_DIR / "knowledge_base_index"
CHROMA_COLLECTION_NAME = "emil_knowledge_base"

# Ollama — a local server, not a Python client library, so no SDK
# dependency is needed for this part (see chromadb's own `rag` extra in
# pyproject.toml for the vector-store side).
#
# The host is overridable via the OLLAMA_HOST_URL environment variable,
# not hardcoded to "localhost": this project runs from two separate
# checkouts on the same machine (a native Windows venv and a WSL2 venv —
# see CLAUDE.md/session history), but Ollama itself only runs once, on
# Windows. When the Streamlit process is the one running on Windows,
# "localhost" is correct. When it's running inside WSL2, "localhost"
# resolves to WSL's own loopback, not Windows' — WSL2 (in the default NAT
# networking mode, confirmed on this machine) can only reach a
# Windows-hosted service bound to 0.0.0.0 via the Windows host's gateway
# IP (`ip route show default` from inside WSL), which is NOT the same
# thing as "localhost" and can change across WSL restarts. Rather than
# hardcode that IP (fragile — confirmed it's specific to this machine's
# current WSL session), the WSL-side environment should export
# OLLAMA_HOST_URL (e.g. in ~/.bashrc: `export
# OLLAMA_HOST_URL=http://$(ip route show default | awk '{print $3}'):11434`
# to pick up the current gateway IP automatically on every shell start).
# Ollama itself must also be bound to 0.0.0.0, not the 127.0.0.1 default,
# for the WSL side to ever reach it at all — see OLLAMA_HOST in Ollama's
# own settings.
OLLAMA_HOST_URL = os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434")
OLLAMA_EMBEDDINGS_URL = f"{OLLAMA_HOST_URL}/api/embeddings"
OLLAMA_GENERATE_URL = f"{OLLAMA_HOST_URL}/api/generate"

# Ollama unloads an idle model after OLLAMA_KEEP_ALIVE (its own default:
# 5 minutes) and reloads it on the next request. That reload includes a
# GPU-discovery step with its own internal watchdog timeout; confirmed
# directly (server log) that this discovery step can itself time out under
# normal conditions on this machine, and when it does, Ollama returns a
# 500 for the request that triggered the reload — even though the model
# then finishes loading a moment later and every following request
# succeeds. So the first embedding/generation call after any idle gap is
# expected to occasionally 500 for reasons outside this app's control; a
# short retry-with-backoff (see indexer.embed() and llm._generate_ollama())
# rides out exactly that window rather than failing the whole operation.
# 5, not 3: a slightly wider safety margin on top of the idle-reload
# retry above — now that report generation is serialized against itself
# (see core/inspections/report_worker.py) rather than several reports
# hitting Ollama at once, retries should rarely need to trigger at all;
# the extra headroom costs nothing on the rare occasions they still do.
OLLAMA_MAX_RETRIES = 5
OLLAMA_RETRY_BACKOFF_SECONDS = 2.0

# nomic-embed-text is the default: small, fast, well-benchmarked for
# English. If the knowledge base is primarily Swedish (or otherwise
# non-English), switch to bge-m3 — a multilingual embedding model — via
# this setting; nothing else needs to change. Whichever model indexed a
# given knowledge base must also be used to embed queries against it later
# (retriever.py), since embedding spaces from different models aren't
# comparable.
DEFAULT_RAG_EMBEDDING_MODEL = "nomic-embed-text"

DEFAULT_RAG_LLM_MODEL = "qwen3:8b"

# core/reporting/llm.py's active backend. "mock" (default) returns a fixed,
# deterministic response with no network call — lets the whole
# retrieve -> analyze -> prompt -> generate chain be verified without
# waiting on Ollama or letting generation's own variability obscure
# whether the orchestration itself is correct. Switch to "ollama" once
# that chain is verified; "cloud" is a stub for a higher-quality API,
# not implemented yet.
DEFAULT_LLM_MODE = "ollama"
VALID_LLM_MODES = ("mock", "ollama", "cloud")

# Chunk size/overlap in words, not characters — simpler to reason about
# for these short spec-sheet/manual-style documents, and avoids cutting
# mid-word. Overlap keeps a sentence that straddles a chunk boundary
# findable from either chunk.
RAG_CHUNK_SIZE_WORDS = 150
RAG_CHUNK_OVERLAP_WORDS = 30

VALID_DOC_TYPES = ("manual", "spec", "incident", "machine_context")

# Reporting is opt-in per component — detection always stands on its own;
# reporting is an optional layer on top (see core/reporting/reporter.py's
# should_generate_report(), the single point pipeline.inspect() calls).
# When off, nothing in core.reporting is ever invoked for that component —
# behavior is identical to before RAG existed. reporting_condition is a
# refinement on top of that on/off switch, not a substitute for it:
# "never" with reporting_enabled on is behaviorally the same as disabled,
# kept as a distinct explicit choice rather than folded into the boolean.
DEFAULT_REPORTING_ENABLED = False
DEFAULT_REPORTING_CONDITION = "on_failed"
VALID_REPORTING_CONDITIONS = ("never", "on_failed", "always", "on_classes")
# JSON list of class names, only consulted when reporting_condition == "on_classes".
DEFAULT_REPORTING_CLASSES = "[]"

# --- Machine context (core/reporting/machine_context) -----------------------
# Which machine parameters matter is per-component data, not fixed schema:
# a toothbrush line cares about temperature/vibration, an optical
# inspection cares about brightness/exposure. Component.machine_parameters
# holds a JSON-encoded list of MachineParameterDef (see
# core/reporting/machine_context/parameters.py) — name, unit, normal
# range, and how a deviation in each direction is phrased as a searchable
# state. Stored on the `components` table like every other per-component
# setting (see config/database.py), but as one JSON column rather than
# fixed per-parameter columns, since the number of parameters varies by
# component. Empty list ("[]") is a valid, unremarkable default — a
# component with no machine-parameter definitions simply has no machine
# context available, the same as a missing reading (see analyzer.py).
DEFAULT_MACHINE_PARAMETERS = "[]"

# Raw time-series machine readings live in the same emil.db as everything
# else, but in their own table, managed directly by
# core/reporting/machine_context/source.py rather than through
# config/database.py's SCHEMA/_MIGRATIONS — this is telemetry data, not a
# per-component setting like the ranges above, so it doesn't fit the
# components-table pattern.
MACHINE_READINGS_TABLE = "machine_readings"

# --- Inspections (core/inspections) -----------------------------------------
# Every inspection's queryable, durable record — verdict, score, defect
# classes, the full report text + provenance (once generated), and the
# input/analyzed/archive lifecycle status. Self-contained schema
# management (see core/inspections/store.py), same pattern as
# machine_readings above: a domain-specific table, not a per-component
# setting, so it doesn't go through config/database.py's SCHEMA.
INSPECTIONS_TABLE = "inspections"

# --- Training runs (core/training_runs) -------------------------------------
# Every training attempt's durable performance record — metrics, confusion
# matrix, per-epoch history, the settings that produced them, and whether it
# succeeded or failed. Self-contained schema management (see
# core/training_runs/store.py), same pattern as INSPECTIONS_TABLE above: a
# domain-specific table, not a per-component setting, so it doesn't go
# through config/database.py's SCHEMA.
TRAINING_RUNS_TABLE = "training_runs"

# --- Evaluation reporting (core/evaluation) ----------------------------------
# Whether train_component() (training/onboard.py) generates and saves
# model_type-aware evaluation artifacts (metrics.json + plots — see
# BaseTrainer.evaluate(), core/base.py) after every successful training run.
# On by default: transparency into how a trained model actually performs is
# the right default for a course/demo project, not an opt-in extra. Off is
# for fast iteration when an operator is deliberately doing several quick
# retrains in a row (e.g. sweeping settings by hand) and doesn't need a full
# report each time — grid_search.py's own sweep trials never generate one
# regardless of this setting (see BaseTrainer.evaluate()'s docstring), so
# this only governs real "train" attempts.
DEFAULT_GENERATE_EVALUATION_REPORT = True

# How long an archived inspection's files are kept before a (not yet
# built) cleanup job would be allowed to delete them. Per-component,
# consistent with every other tunable in this project — a component
# producing far more traffic than another may reasonably want a shorter
# retention window. Only the setting exists for now; no cleanup job reads
# it yet (see core/inspections/lifecycle.py's module docstring).
DEFAULT_INSPECTION_RETENTION_DAYS = 90

# How an approved verdict is handled once it's recorded. A production line
# generates far more approved than failed inspections, and an operator
# reviewing history is almost always most interested in failed ones — but
# an approved record must never become unreachable, since it may itself be
# a false negative (a real defect the model missed), and false negatives
# are usually the more dangerous failure mode. All three modes keep every
# approved record permanently queryable; they only differ in whether a
# human has to look at it before it's eligible for archiving:
# - "keep_visible": no special handling — shown in the Inspection Station
#   like any other record (today's behavior before this setting existed).
# - "hide_from_default_view": approved records are hidden from the
#   station's default view but always reachable via its approved filter —
#   the recommended default for a busy line.
# - "auto_acknowledge": approved records are acknowledged automatically
#   (see core/inspections/orchestrator.py) — no human has to click
#   Acknowledge on a passing unit — but archiving still only happens on
#   the normal retention schedule, not immediately, so there's still a
#   window to review one before it's filed away.
VALID_APPROVED_HANDLING_MODES = ("keep_visible", "hide_from_default_view", "auto_acknowledge")
DEFAULT_APPROVED_HANDLING = "hide_from_default_view"

# Whether/how a human-verified correction (see core/inspections/store.py's
# verify()) gets copied into this component's actual training/ data —
# where it becomes permanent and gets picked up by every future training
# run automatically, same as original onboarding data. A deliberate,
# per-component choice, not a fixed behavior: a single mis-annotated
# correction copied in unreviewed can quietly poison the training set,
# so how much human judgment sits between "verified" and "trained on" is
# a real policy decision, not a technical default.
# - "off": corrections accumulate as verified (pending) but are NEVER
#   copied into training/ automatically — training data only changes if
#   a human explicitly does something about it later (e.g. by switching
#   this to manual_review/automatic first). Safest, most inert option;
#   training data stays perfectly stable regardless of correction volume.
# - "manual_review": corrections are offered at the existing incorporate-
#   before-training selection step (see the Onboard page's Train
#   section) — a human picks exactly which pending corrections go in
#   this round; anything left unchecked stays pending and is offered
#   again next time. This is the real quality gate: a bad annotation
#   gets caught by a human before it ever reaches training/, the same
#   reasoning that makes auto_acknowledge above an opt-in rather than a
#   default.
# - "automatic": every pending verified correction is copied into
#   training/ without a human in the loop, trusting the correction flow
#   completely. Fastest path from correction to improved model; a
#   deliberate opt-in for a component whose correction flow is trusted,
#   not a default anyone should land on by accident.
VALID_VERIFIED_CORRECTION_POLICIES = ("off", "manual_review", "automatic")
DEFAULT_VERIFIED_CORRECTION_POLICY = "manual_review"

# Whether a component is in active use, deactivated, or soft-deleted — a
# separate axis from `status` (created/training/ready/failed, which
# tracks TRAINING progress) entirely. Every registry-driven iteration
# (the watcher's input/ scan, RAG's index_all(), the Inspect page's
# component picker) filters to "active" only; a management view (Onboard
# page) shows active+inactive but hides deleted; only the trash view
# shows deleted. See config/registry.py's deactivate()/reactivate()/
# soft_delete()/restore() for the transitions between them, and
# core/component_deletion.py for what (if anything) actually happens to
# a component's data at each stage:
# - "active" (default): normal, in-use component.
# - "inactive": deliberately paused — reversible, zero data impact.
#   Nothing is touched; only excluded from active-use iteration.
# - "deleted": soft-deleted, sitting in the trash. Still zero data
#   impact — everything (models, training data, inspection history,
#   knowledge docs, ChromaDB chunks) is untouched and fully restorable
#   via restore(). Only core/component_deletion.py's
#   permanently_delete_component() actually removes anything, and only
#   for a component already in this state.
VALID_LIFECYCLE_STATUSES = ("active", "inactive", "deleted")
DEFAULT_LIFECYCLE_STATUS = "active"

# How long a soft-deleted component sits in the trash before it becomes
# eligible for permanent deletion by core/component_deletion.py's
# cleanup_expired_soft_deleted_components() — a global policy, not
# per-component (unlike inspection_retention_days, there's no clear
# reason this should vary by component). Not run automatically — the
# Onboard page's trash view exposes it as an explicit "Run cleanup now"
# button, same "no destructive action without a deliberate click"
# posture as core/inspections/retention.py's cleanup. A component can
# also be permanently deleted immediately from the trash, bypassing this
# window entirely, for an operator who's already certain.
DEFAULT_COMPONENT_DELETION_RETENTION_DAYS = 30

# --- Cascade framework (core/cascade) ----------------------------------------
# Generic multi-stage frame classification, built on top of the same
# registry-driven-dispatch principle as registry_factory.py, one level up:
# a coarse classifier (any (modality, model_type) — see
# core/classification/resnet_coarse for the first one) narrows a frame to a
# category; a PER-COMPONENT mapping (this setting, below) says which
# categories activate which specialist (face recognition is the first
# one; not person-specific by design — see specialist_registry.py);
# core/cascade/policy.py maps the specialist's identified identity to a
# reaction policy. None of these three stages are hardcoded branches
# anywhere in core/cascade/pipeline.py — each is its own data structure,
# looked up, never branched on.

# Per-component: which coarse categories activate a specialist, and which
# one. JSON dict, e.g. '{"human": "face"}' — see
# core/cascade/specialist_registry.py's parse_category_specialists()/
# serialize_category_specialists() and its own DEFAULT_CATEGORY_SPECIALISTS
# (this setting's default value, kept there rather than duplicated here
# since that module already needs the mapping at runtime). A category
# absent from this mapping (every category except "human", by default) is
# a normal, valid "detect and report, no further identification" outcome
# — not an error, not a gap; see pipeline.py. Editable from the Onboard
# page with no code change: this is exactly the mechanism a future
# vehicle -> car_classifier activation would use once that specialist
# exists, not a different one.
DEFAULT_CASCADE_CATEGORY_SPECIALISTS = '{"human": "face"}'

# Face-recognition specialist (core/cascade/specialists/face) match
# threshold, in embedding L2 distance (facenet-pytorch's InceptionResnetV1,
# vggface2 weights — 512-dim embeddings). Below this distance to a known
# individual's stored embedding: a match. At or above it, for every known
# individual: "unknown" — the same first-class, no-match-is-still-a-result
# outcome as `resnet_confidence_threshold`'s "uncertain" above, not an
# error. Not per-component (the cascade isn't scoped to one Component the
# way anomaly inspection is) — one global tunable, same spirit as
# DEFAULT_THRESHOLD_PERCENTILE. 0.9 is a commonly-cited starting point for
# this embedding space; tune against your own known-individuals set if
# false-accepts/false-rejects show up in practice — once at least two
# individuals are registered with a couple of photos each,
# specialists/face/store.py's calibration_stats() computes the actual
# observed intra-person (same person, different photos) and inter-person
# (different people) embedding-distance distributions this threshold
# should sit between, the same "calibrate against an observed
# distribution, don't guess" principle core/evaluation applies to the
# unsupervised image methods' anomaly threshold.
DEFAULT_FACE_MATCH_DISTANCE_THRESHOLD = 0.9

# Where the "save_frame" reaction-policy action (core/cascade/policy_executor)
# writes a frame — scoped under its own directory, deliberately not mixed
# into any component's own analyzed/ folder: a saved cascade frame belongs
# to an identity (or "unknown"), not to a component, and may contain a
# consenting individual's face (see specialists/face/store.py's module
# docstring on why the known-individuals table itself is opt-in only).
CASCADE_SAVED_FRAMES_DIR = DATA_DIR / "cascade" / "saved_frames"

# Per-component: Kafka connection for a continuous cascade stream (see
# emil_ml/cascade_stream) — empty string means "not configured yet", which
# the stream process and the Cascade Stream page both treat as a hard stop,
# not a fallback to some default broker. Only meaningful for coco_detector
# components (same "ignored by other model_types" convention as
# coco_confidence_threshold above).
DEFAULT_CASCADE_STREAM_KAFKA_BOOTSTRAP_SERVERS = ""
DEFAULT_CASCADE_STREAM_KAFKA_TOPIC = ""

# Per-component: throttle for how often a frame from a continuous source
# (Kafka or an uploaded video) is actually run through the cascade — "check
# at most once every N seconds" rather than every single frame, since a
# real camera feed (or a video's own frame rate) is almost always far
# denser than an object's presence needs to be re-confirmed. Compared
# against each Frame's own position_seconds (core/cascade/frame_sources.py),
# not wall-clock time, so this means the same thing for a live Kafka feed
# and a video decoding faster than real-time alike — see
# core/cascade/stream_processor.py's should_sample(). 1.0 (at most 1
# frame/sec) is a reasonable default for "doesn't need to check that many
# frames per second."
DEFAULT_CASCADE_STREAM_SAMPLE_RATE_SECONDS = 1.0

# Every frame actually processed by a cascade stream is thumbnailed here —
# unconditional, unlike CASCADE_SAVED_FRAMES_DIR above (which is
# identity-scoped and only written when a reaction policy's own
# "save_frame" action fires). No retention/cleanup job exists for this
# directory yet — a long-running Kafka consumer will grow it without bound;
# a future addition, not handled here.
CASCADE_STREAM_FRAMES_DIR = DATA_DIR / "cascade" / "stream_frames"

# --- Cascade live stream (emil_ml/cascade_stream, core/cascade/stream_processor+frame_sources) ---
# Continuous operation of the cascade above — either a standalone process
# consuming a Kafka topic (emil_ml/cascade_stream, started the same way
# emil-watcher is: a terminal command, never launched by Streamlit — see
# that package's own module docstring) or a synchronous, bounded pass over
# an uploaded video file (run inline from the Cascade Stream page). Both
# paths share core/cascade/stream_processor.py's per-frame logic and
# core/cascade/stream_store.py's persistence; only where the frame comes
# from differs.

# How often (seconds) the Kafka consumer process writes a heartbeat row —
# i.e. how quickly the page's "is this actually running right now" status
# can go stale-but-still-marked-running before it's treated as stopped/
# crashed. Frame RESULTS are still persisted on every processed frame
# regardless of this — it only governs the separate liveness ping, so it
# can stay coarse without losing any actual results.
CASCADE_STREAM_HEARTBEAT_INTERVAL_SECONDS = 5.0

# How long since the last heartbeat before the page stops trusting a
# status='running' row and treats the stream as stopped/crashed instead —
# deliberately several heartbeat intervals, not one, so a single slow tick
# doesn't flash a false "stopped" while the process is still fine.
CASCADE_STREAM_HEARTBEAT_STALE_SECONDS = 20.0

# Kafka Consumer.poll() timeout, seconds — how long one poll() call blocks
# waiting for a message before returning and looping again. Also this
# loop's own responsiveness to a stop request, since that's only checked
# between poll() calls.
CASCADE_STREAM_KAFKA_POLL_TIMEOUT_SECONDS = 1.0

# How often (seconds) the Cascade Stream page's live-results feed re-polls
# the database (st.fragment(run_every=...), the same cross-process-safe
# pattern app/pages/1_inspect.py's _render_pending_report() already uses
# for a report streaming in — core/inspections/progress.py's in-memory
# dict, by contrast, is explicitly single-process only and could never work
# here, since the Kafka consumer is a separate OS process). A fixed UI
# cadence, not a per-component setting.
DEFAULT_CASCADE_STREAM_UI_POLL_SECONDS = 2.0

# Cascade stream run/result history lives in its own self-contained schema
# (see core/cascade/stream_store.py), same "domain-specific table, not a
# per-component setting" pattern as INSPECTIONS_TABLE/TRAINING_RUNS_TABLE
# above — a run's frame-by-frame results don't fit components' SCHEMA any
# more than an inspection's per-detection boxes do.
CASCADE_STREAM_RUNS_TABLE = "cascade_stream_runs"
CASCADE_STREAM_RESULTS_TABLE = "cascade_stream_results"

# --- Folder watcher (emil_ml/watcher) ----------------------------------------
# A standalone process, independent of Streamlit — see watcher/service.py's
# module docstring for the full design. These three numbers are its only
# real tuning knobs, all in seconds.

# A file is considered fully written once its size hasn't changed across
# this many consecutive checks, WATCHER_STABILITY_CHECK_INTERVAL_SECONDS
# apart — never act on a file that might still be mid-write (a truncated
# read produces silent or cryptic errors, the most important robustness
# detail for a real camera feed). 2 checks / 1s apart means a file sitting
# idle for >=1s is treated as done — long enough to ride out a slow local
# write, short enough not to noticeably delay processing.
WATCHER_STABILITY_CHECK_INTERVAL_SECONDS = 1.0
WATCHER_STABILITY_REQUIRED_CHECKS = 2

# The watchdog-events path is unreliable enough on its own — especially
# over a network share a production camera might write to — that a
# periodic full rescan of every component's input/ is a required safety
# net, not an optional extra. Also doubles as the mechanism a newly
# onboarded component's input/ starts being watched without a restart
# (see service.py's _sync_watches()). 10s balances catching a missed
# event reasonably quickly against constantly re-listing every input/ dir.
WATCHER_POLL_INTERVAL_SECONDS = 10.0

# --- Logging (config/logging_config.py) --------------------------------------
# One central log directory for the whole app, not one per component —
# see logging_config.py's own module docstring for why. Alongside
# emil.db at the project root's data/ dir, same convention
# CHROMA_PERSIST_DIR already uses for "derived/generated, not source of
# truth" project-wide state — not hardcoded elsewhere, so moving
# DATA_DIR moves this with it.
#
# One file per calendar day (log<YYYYMMDD>.txt, e.g. log20260809.txt) —
# see logging_config.py's _DailyFileHandler: a long-lived process (the
# watcher, in particular, or a Streamlit server left running) rolls over
# to a fresh dated file the moment the date changes, without needing a
# restart. Each day's events stay chronologically together in one file
# without needing size-based rotation's arbitrary split points, and old
# days are trivial to find or archive by filename alone.
LOG_DIR = DATA_DIR / "logs"
LOG_FILE_PREFIX = "log"
LOG_FILE_SUFFIX = ".txt"
LOG_FILE_DATE_FORMAT = "%Y%m%d"

# INFO by default; override per-process via the EMIL_LOG_LEVEL env var
# (e.g. `EMIL_LOG_LEVEL=DEBUG streamlit run app/streamlit_app.py`) — same
# "env var overrides a hardcoded default" pattern OLLAMA_HOST_URL already
# uses, since Streamlit and the verify scripts have no CLI flag of their
# own to carry a --log-level through, unlike the watcher (which exposes
# its own --log-level and passes it to configure_logging() directly
# rather than going through this env var).
DEFAULT_LOG_LEVEL = os.environ.get("EMIL_LOG_LEVEL", "INFO")

COMPONENTS_DIR.mkdir(parents=True, exist_ok=True)
CASCADE_SAVED_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
CASCADE_STREAM_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
