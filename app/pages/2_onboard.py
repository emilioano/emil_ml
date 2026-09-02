"""Onboard page: register a new component, provide training data, configure
settings, and train.

Pure view layer — all logic is delegated to emil_ml.training.onboard. Most
of this page never branches on model_type (advanced settings are shown
unconditionally — a method simply ignores the ones it doesn't use, same
reasoning throughout the project). The one deliberate exception is YOLO's
creation/annotation flow: its data model (annotated images + boxes, produced
via one of three distinct paths) is fundamentally different from the
approved/failed file convention every other method uses, so this page shows
a different UI section for it. That's a presentation-layer decision inside
onboarding, which is explicitly where per-method logic is expected to live —
it does not reach into pipeline/registry_factory/watcher, which stay
completely method-agnostic.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd
import altair as alt
import streamlit as st
from PIL import Image

# app/ (this file's parent's parent) holds shared_yolo_canvas.py, a
# sibling of pages/ rather than part of the emil_ml package (it's view-
# layer, not business logic — see that module's own docstring). Explicit
# sys.path bootstrap rather than relying on Streamlit's own working-
# directory behavior, so the import resolves identically whether this
# page runs via `streamlit run`, AppTest, or any other entry point.
_APP_DIR = Path(__file__).resolve().parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from shared_yolo_canvas import render_yolo_box_canvas  # noqa: E402 (after sys.path bootstrap above)

from emil_ml.config.logging_config import configure_logging
from emil_ml.config.registry import SETTINGS_DEFAULTS, ComponentRegistry
from emil_ml.config.settings import (
    DEFAULT_COMPONENT_DELETION_RETENTION_DAYS,
    DEFAULT_FACE_MATCH_DISTANCE_THRESHOLD,
    VALID_APPROVED_HANDLING_MODES,
    VALID_CLASS_WEIGHT_STRATEGIES,
    VALID_CLASSIFIER_BASE_MODELS,
    VALID_CLASSIFIER_POOLING_TYPES,
    VALID_ISOLATION_FOREST_CONTAMINATION_OPTIONS,
    VALID_PATCHCORE_BACKBONES,
    VALID_REPORTING_CONDITIONS,
    VALID_SCORE_METHODS,
    VALID_VERIFIED_CORRECTION_POLICIES,
    VALID_YOLO_DECISION_RULES,
    VALID_YOLO_MODEL_VARIANTS,
    VALID_YOLO_OPTIMIZERS,
)
from emil_ml.core import component_deletion, registry_factory
from emil_ml.core.cascade import pipeline as cascade_pipeline
from emil_ml.core.cascade import policy_store, specialist_registry
from emil_ml.core.cascade.categories import (
    CATEGORY_ANIMAL,
    CATEGORY_HUMAN,
    CATEGORY_OTHER,
    CATEGORY_UNCERTAIN,
    CATEGORY_VEHICLE,
)
from emil_ml.core.cascade.policy import DEFAULT_PRIORITY, VALID_ACTIONS, VALID_PRIORITIES
from emil_ml.core.cascade.specialists.face import store as face_store
from emil_ml.core.cascade.specialists.face.predictor import detect_all_faces
from emil_ml.core.detection.yolo import annotation as yolo_annotation
from emil_ml.core.inspections import retention
from emil_ml.core.reporting.knowledge import indexer as reporting_indexer
from emil_ml.core.reporting.machine_context.parameters import (
    MachineParameterDef,
    parse_machine_parameters,
    serialize_machine_parameters,
)
from emil_ml.core.search import grid_search
from emil_ml.core.training_runs import store as training_runs_store
from emil_ml.training import onboard
from emil_ml.utils.paths import for_component

configure_logging()

st.set_page_config(page_title="EMIL Lab — Onboard", page_icon="🧩", layout="wide")
st.title("Onboard a new component")

registry = ComponentRegistry()

MODEL_TYPE_OPTIONS = {
    "Autoencoder": "autoencoder",
    "Classifier": "classifier",
    "YOLO (object detection)": "yolo",
    "PatchCore (unsupervised, patch-level)": "patchcore",
    "Isolation Forest (unsupervised, CNN embeddings)": "isolation_forest",
    "Object & face cascade (COCO detector)": "coco_detector",
}
AVAILABLE_MODEL_TYPES = {"autoencoder", "classifier", "yolo", "patchcore", "isolation_forest"}
MODEL_TYPE_DESCRIPTIONS = {
    "autoencoder": (
        "**Autoencoder (unsupervised)** — trains on approved files only, needs no defect examples. "
        "Good when defects are unpredictable/rare and the anomaly is spatially large relative to "
        "the image. Weak when defects are small and localized — a global reconstruction-error "
        "average gets diluted by an otherwise-correct image (this is what happened on real "
        "toothbrush defect data: good/defective barely separated). Try the 'local_max' score "
        "method under Advanced settings if you suspect this."
    ),
    "classifier": (
        "**Supervised classifier (CNN, transfer learning)** — trains on both approved AND failed "
        "files, learns to tell them apart directly. Good with a reasonable number of labeled "
        "examples of both classes. Weak with very little data — can collapse toward one class "
        "(e.g. every real defect caught, but most approved files wrongly flagged too, which looks "
        "like great recall while being close to useless). Requires labeled defect examples — the "
        "autoencoder doesn't."
    ),
    "yolo": (
        "**YOLO (object detection)** — localizes and draws a box around specific objects/defects "
        "instead of judging the whole image. Good when a defect (or expected object) is small and "
        "localized — a foreign object, a specific visible flaw — since it doesn't need to detect a "
        "change across the whole image, just find the thing. This is exactly the case where the "
        "autoencoder and classifier both struggle. Weak/more work: needs bounding-box-annotated "
        "data, not just images sorted into folders — see the three annotation options below."
    ),
    "patchcore": (
        "**PatchCore (unsupervised, patch-level)** — like the autoencoder, trains on approved "
        "files only, needs no defect examples. Unlike the autoencoder, it never judges the whole "
        "image at once: a frozen pretrained CNN backbone extracts a grid of local patch features, "
        "and a memory bank of normal patches (built once from approved images) is what 'normal' "
        "is defined by — a new image's score is how far its own patches sit from their nearest "
        "neighbors in that memory bank. Good when a defect is small and localized — exactly what "
        "the autoencoder and classifier both struggle with — and shows *where* the anomaly is via "
        "a heatmap, without ever needing a single annotated bounding box (see it on the Inspect "
        "page). Weak when normal images vary a lot on their own (lighting, pose, background "
        "clutter) — that natural variation can trigger as many false alarms as a real defect "
        "would. Heavier dependency than the rest of EMIL Lab (PyTorch via anomalib, a separate "
        "optional install — see pyproject.toml's `patchcore` extra)."
    ),
    "isolation_forest": (
        "**Isolation Forest (unsupervised)** — like the autoencoder and PatchCore, trains on "
        "approved files only, needs no defect examples. Instead of working on pixels, it reuses "
        "the same frozen CNN backbone the diagnostics page's UMAP/PCA plots use to reduce each "
        "image to a single embedding vector, then fits an Isolation Forest on the approved set's "
        "vectors — a new image is flagged if its embedding is unusually easy to isolate from "
        "that fitted forest. Simple and fast to train (seconds, no epochs). Good when a defect "
        "changes the image's overall character enough to shift its whole-image embedding — check "
        "the diagnostics page first to see whether approved/failed embeddings actually separate "
        "for your data. Weak when the defect is small and localized — like the autoencoder and "
        "classifier, whole-image pooling can wash out a small local anomaly; PatchCore is the "
        "better fit for that case."
    ),
    "coco_detector": (
        "**Object & face cascade** — not a defect detector: a frozen, pretrained COCO-YOLO that "
        "finds people, animals, and vehicles in a frame (with boxes), and — for each detected "
        "person — runs face recognition against a database of consenting, named individuals "
        "you register below. No training data, no annotation: it's ready to use immediately. "
        "See the 'Cascade: object & face recognition' section further down the page once created."
    ),
}

SCORE_METHOD_OPTIONS = list(VALID_SCORE_METHODS)
BASE_MODEL_OPTIONS = list(VALID_CLASSIFIER_BASE_MODELS)
POOLING_OPTIONS = list(VALID_CLASSIFIER_POOLING_TYPES)
CLASS_WEIGHT_OPTIONS = list(VALID_CLASS_WEIGHT_STRATEGIES)
YOLO_MODEL_VARIANT_OPTIONS = list(VALID_YOLO_MODEL_VARIANTS)
YOLO_DECISION_RULE_OPTIONS = list(VALID_YOLO_DECISION_RULES)
YOLO_OPTIMIZER_OPTIONS = list(VALID_YOLO_OPTIMIZERS)
PATCHCORE_BACKBONE_OPTIONS = list(VALID_PATCHCORE_BACKBONES)
ISOLATION_FOREST_CONTAMINATION_OPTIONS = list(VALID_ISOLATION_FOREST_CONTAMINATION_OPTIONS)

UPLOAD_TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]

# Every coarse category the cascade's Step 1 (core/detection/yolo_coco) can
# produce — "human" first since it's the one with a default specialist
# activated (see specialist_registry.DEFAULT_CATEGORY_SPECIALISTS); the
# order here is just the order the category -> specialist configuration
# form below lists them in.
ALL_CASCADE_CATEGORIES = [CATEGORY_HUMAN, CATEGORY_ANIMAL, CATEGORY_VEHICLE, CATEGORY_OTHER, CATEGORY_UNCERTAIN]
NO_SPECIALIST_OPTION = "(none — detect & report only)"


def _render_advanced_settings(defaults: dict, key_prefix: str) -> dict:
    """Render every method's advanced settings, pre-filled from `defaults`. Returns chosen values."""
    st.caption("Shared")
    early_stopping_patience = st.number_input(
        "Early stopping patience",
        min_value=0,
        value=int(defaults["early_stopping_patience"]),
        key=f"{key_prefix}_early_stopping_patience",
        help=(
            "Stop a training phase once its monitored loss hasn't improved for this many "
            "consecutive epochs, restoring the best epoch's weights instead of the last one. "
            "Guards against overfitting an already-small dataset by training past the point "
            "it's actually helping. 0 disables it and always runs the full epoch budget."
        ),
    )

    st.caption("Autoencoder")
    score_method = st.selectbox(
        "Score method",
        SCORE_METHOD_OPTIONS,
        index=SCORE_METHOD_OPTIONS.index(defaults["score_method"]),
        key=f"{key_prefix}_score_method",
        help=(
            "How the reconstruction-error score is computed. 'global_mean' (default): average "
            "error over the whole image. 'local_max': the single worst-reconstructed pixel — "
            "more sensitive to a small, localized defect that global_mean would dilute across "
            "an otherwise correctly-reconstructed image."
        ),
    )
    threshold_percentile = st.number_input(
        "Threshold percentile",
        min_value=50.0,
        max_value=100.0,
        step=0.5,
        value=float(defaults["threshold_percentile"]),
        key=f"{key_prefix}_threshold_percentile",
        help=(
            "Percentile of the approved reconstruction-error distribution used as the anomaly "
            "threshold. Higher = stricter default (fewer approved flagged, but may catch fewer "
            "real defects); lower = more sensitive."
        ),
    )

    st.caption("Classifier")
    base_model = st.selectbox(
        "Base model",
        BASE_MODEL_OPTIONS,
        index=BASE_MODEL_OPTIONS.index(defaults["base_model"]),
        key=f"{key_prefix}_base_model",
        help="Pretrained backbone used for transfer learning.",
    )
    pooling = st.selectbox(
        "Pooling",
        POOLING_OPTIONS,
        index=POOLING_OPTIONS.index(defaults["pooling"]),
        key=f"{key_prefix}_pooling",
        help=(
            "How the base model's feature map is pooled before the classification head. "
            "'max' keeps the strongest local activation — in theory better for a small, "
            "localized defect, but needs more data to learn reliably (it only backprops "
            "through a single spatial location per step). 'average' (default) smooths over "
            "the whole feature map, and has outperformed 'max' on real data tested so far."
        ),
    )
    class_weight_strategy = st.selectbox(
        "Class weight strategy",
        CLASS_WEIGHT_OPTIONS,
        index=CLASS_WEIGHT_OPTIONS.index(defaults["class_weight_strategy"]),
        key=f"{key_prefix}_class_weight_strategy",
        help=(
            "'balanced' (default): weight each class inversely to its frequency, so the "
            "majority class doesn't dominate the loss. 'none': equal weight — try this if "
            "'balanced' seems to be overcorrecting (e.g. collapsing toward the minority class)."
        ),
    )
    augmentation_strength = st.number_input(
        "Augmentation strength",
        min_value=0.0,
        max_value=0.5,
        step=0.01,
        value=float(defaults["augmentation_strength"]),
        key=f"{key_prefix}_augmentation_strength",
        help=(
            "Strength of random rotation/zoom/brightness applied during training (horizontal "
            "flip is always on regardless). Kept mild by default — aggressive augmentation can "
            "distort or crop away a subtle, localized defect more often than it helps the model "
            "generalize. Lower it (even to 0) if a defect might be getting destroyed before the "
            "model ever sees it clearly."
        ),
    )
    fcol1, fcol2, fcol3 = st.columns(3)
    fine_tune_epochs = fcol1.number_input(
        "Fine-tune epochs",
        min_value=1,
        value=int(defaults["fine_tune_epochs"]),
        key=f"{key_prefix}_fine_tune_epochs",
        help="Epochs for the fine-tuning phase, which runs after head-only training.",
    )
    fine_tune_learning_rate = fcol2.number_input(
        "Fine-tune LR",
        min_value=0.0,
        step=0.00001,
        format="%.6f",
        value=float(defaults["fine_tune_learning_rate"]),
        key=f"{key_prefix}_fine_tune_learning_rate",
        help="Learning rate during fine-tuning — kept much lower than head training so it adapts features instead of destroying them.",
    )
    fine_tune_unfreeze_layers = fcol3.number_input(
        "Unfreeze layers",
        min_value=0,
        value=int(defaults["fine_tune_unfreeze_layers"]),
        key=f"{key_prefix}_fine_tune_unfreeze_layers",
        help="Number of the base model's top layers to unfreeze during fine-tuning. 0 = don't fine-tune the base at all.",
    )
    st.caption(
        "Fine-tuning is always evaluated against head-only training and only kept if it actually "
        "validates better (by balanced accuracy), so raising these is safe to experiment with — "
        "it can't make results worse than training without it."
    )

    st.caption("YOLO")
    ycol1, ycol2 = st.columns(2)
    yolo_model_variant = ycol1.selectbox(
        "YOLO model variant",
        YOLO_MODEL_VARIANT_OPTIONS,
        index=YOLO_MODEL_VARIANT_OPTIONS.index(defaults["yolo_model_variant"]),
        key=f"{key_prefix}_yolo_model_variant",
        help="Pretrained checkpoint fine-tuned from. 'n' (nano) is smallest/fastest; 's' (small) trades speed for more capacity.",
    )
    decision_rule = ycol2.selectbox(
        "Decision rule",
        YOLO_DECISION_RULE_OPTIONS,
        index=YOLO_DECISION_RULE_OPTIONS.index(defaults["decision_rule"]),
        key=f"{key_prefix}_decision_rule",
        help=(
            "'presence': looking for something that shouldn't be there (a defect, a foreign "
            "object) — found = failed. 'absence': looking for something that should be there "
            "(a required part) — not found = failed."
        ),
    )
    ycol3, ycol4 = st.columns(2)
    yolo_mosaic = ycol3.slider(
        "Mosaic augmentation",
        min_value=0.0,
        max_value=1.0,
        value=float(defaults["yolo_mosaic"]),
        step=0.05,
        key=f"{key_prefix}_yolo_mosaic",
        help=(
            "Probability that training splices 4 images into one composite. Defaults to 0 "
            "(disabled) — with the small, few-dozen-image datasets this app is built for, "
            "mosaic can slice an already-scarce small/localized defect apart across the "
            "composite instead of helping the model generalize. Worth raising if a much "
            "larger annotated dataset is available."
        ),
    )
    yolo_class_loss_weight = ycol4.slider(
        "Class loss weight",
        min_value=0.0,
        max_value=4.0,
        value=float(defaults["yolo_class_loss_weight"]),
        step=0.1,
        key=f"{key_prefix}_yolo_class_loss_weight",
        help=(
            "Ultralytics' `cls` loss gain — how strongly training penalizes classification "
            "mistakes relative to box-position mistakes. Not the same idea as the classifier's "
            "class_weight_strategy: with a single 'defect' class this mostly rebalances "
            "classification-vs-localization priority, not per-class inverse-frequency "
            "weighting. Ultralytics' own default is 0.5."
        ),
    )
    yolo_augmentation_strength = st.slider(
        "Geometric/color augmentation strength",
        min_value=0.0,
        max_value=1.0,
        value=float(defaults["yolo_augmentation_strength"]),
        step=0.05,
        key=f"{key_prefix}_yolo_augmentation_strength",
        help=(
            "Strength of rotation/translation/scaling/shear/hue/saturation/value jitter "
            "applied during training, scaled from a moderate baseline at 1.0. Horizontal flip "
            "is left on regardless — it can't destroy a defect's visibility, only mirror its "
            "position. Defaults to 0 (disabled), same reasoning as mosaic above: with only a "
            "few dozen images, jitter can just as easily obscure a small defect as help the "
            "model generalize."
        ),
    )
    ycol5, ycol6 = st.columns(2)
    yolo_optimizer = ycol5.selectbox(
        "Optimizer",
        YOLO_OPTIMIZER_OPTIONS,
        index=YOLO_OPTIMIZER_OPTIONS.index(defaults["yolo_optimizer"]),
        key=f"{key_prefix}_yolo_optimizer",
        help=(
            "'auto' (default) picks both the optimizer *and* its learning rate/momentum "
            "itself based on the model and dataset size — and in doing so, ignores the "
            "learning rate set below entirely (Ultralytics logs this: \"'optimizer=auto' "
            "found, ignoring 'lr0=...'\"). Pick a fixed optimizer (e.g. 'SGD' or 'AdamW') if "
            "you want the learning rate below to actually take effect."
        ),
    )
    yolo_learning_rate = ycol6.number_input(
        "Learning rate (lr0)",
        min_value=0.0,
        step=0.0001,
        format="%.5f",
        value=float(defaults["yolo_learning_rate"]),
        key=f"{key_prefix}_yolo_learning_rate",
        help=(
            "Ultralytics' initial learning rate. Only takes effect if Optimizer above is set "
            "to something other than 'auto' — see its help text. Ultralytics' own default is "
            "0.01."
        ),
    )

    st.caption("PatchCore")
    pcol1, pcol2, pcol3 = st.columns(3)
    patchcore_backbone = pcol1.selectbox(
        "Backbone",
        PATCHCORE_BACKBONE_OPTIONS,
        index=PATCHCORE_BACKBONE_OPTIONS.index(defaults["patchcore_backbone"]),
        key=f"{key_prefix}_patchcore_backbone",
        help=(
            "Frozen pretrained CNN patch features are extracted from — never fine-tuned, "
            "PatchCore only builds a memory bank from its features. 'wide_resnet50_2' "
            "(default) is anomalib's own default and the most benchmarked on MVTec-style "
            "data; 'resnet18' trades some accuracy for a much smaller/faster memory bank — "
            "worth trying on a slow machine or a very small dataset."
        ),
    )
    patchcore_coreset_sampling_ratio = pcol2.slider(
        "Coreset sampling ratio",
        min_value=0.01,
        max_value=1.0,
        value=float(defaults["patchcore_coreset_sampling_ratio"]),
        step=0.01,
        key=f"{key_prefix}_patchcore_coreset_sampling_ratio",
        help=(
            "Fraction of extracted normal-image patch features kept in the memory bank after "
            "subsampling. Lower = smaller/faster memory bank, at some risk of losing rare-but-"
            "normal patch patterns (more false alarms on unusual-but-fine regions); higher = "
            "more faithful memory bank, slower nearest-neighbor search at inference."
        ),
    )
    patchcore_num_neighbors = pcol3.number_input(
        "Num neighbors",
        min_value=1,
        value=int(defaults["patchcore_num_neighbors"]),
        key=f"{key_prefix}_patchcore_num_neighbors",
        help="How many nearest neighbors in the memory bank a test patch is compared against when computing its anomaly score.",
    )

    st.caption("Isolation Forest")
    icol1, icol2 = st.columns(2)
    isolation_forest_n_estimators = icol1.number_input(
        "Number of trees",
        min_value=10,
        value=int(defaults["isolation_forest_n_estimators"]),
        key=f"{key_prefix}_isolation_forest_n_estimators",
        help="How many isolation trees make up the forest. More trees give a more stable score at the cost of slower training/inference.",
    )
    isolation_forest_contamination = icol2.selectbox(
        "Contamination",
        ISOLATION_FOREST_CONTAMINATION_OPTIONS,
        index=ISOLATION_FOREST_CONTAMINATION_OPTIONS.index(str(defaults["isolation_forest_contamination"])),
        key=f"{key_prefix}_isolation_forest_contamination",
        help=(
            "Expected fraction of the training set that's actually anomalous — directly sets "
            "the anomaly threshold. 'auto' (default) uses the forest's own internal offset, "
            "which behaves like a small contamination. Raise this if too many approved images "
            "end up flagged; lower it if real defects are slipping through."
        ),
    )
    icol3, icol4 = st.columns(2)
    isolation_forest_max_features = icol3.slider(
        "Max features",
        min_value=0.1,
        max_value=1.0,
        value=float(defaults["isolation_forest_max_features"]),
        step=0.05,
        key=f"{key_prefix}_isolation_forest_max_features",
        help="Fraction of embedding dimensions randomly sampled per tree split. Lower values increase tree diversity, at some risk of missing the dimensions that actually separate normal from anomalous.",
    )
    isolation_forest_standardize = icol4.checkbox(
        "Standardize embeddings",
        value=bool(defaults["isolation_forest_standardize"]),
        key=f"{key_prefix}_isolation_forest_standardize",
        help="Scale embedding dimensions to zero mean/unit variance before fitting (recommended — Isolation Forest's random splits are sensitive to features being on very different scales). The same scaler is reused at inference time.",
    )

    return {
        "early_stopping_patience": int(early_stopping_patience),
        "score_method": score_method,
        "threshold_percentile": float(threshold_percentile),
        "base_model": base_model,
        "pooling": pooling,
        "class_weight_strategy": class_weight_strategy,
        "augmentation_strength": float(augmentation_strength),
        "fine_tune_epochs": int(fine_tune_epochs),
        "fine_tune_learning_rate": float(fine_tune_learning_rate),
        "fine_tune_unfreeze_layers": int(fine_tune_unfreeze_layers),
        "yolo_model_variant": yolo_model_variant,
        "decision_rule": decision_rule,
        "yolo_mosaic": float(yolo_mosaic),
        "yolo_class_loss_weight": float(yolo_class_loss_weight),
        "yolo_augmentation_strength": float(yolo_augmentation_strength),
        "yolo_optimizer": yolo_optimizer,
        "yolo_learning_rate": float(yolo_learning_rate),
        "patchcore_backbone": patchcore_backbone,
        "patchcore_coreset_sampling_ratio": float(patchcore_coreset_sampling_ratio),
        "patchcore_num_neighbors": int(patchcore_num_neighbors),
        "isolation_forest_n_estimators": int(isolation_forest_n_estimators),
        "isolation_forest_contamination": isolation_forest_contamination,
        "isolation_forest_max_features": float(isolation_forest_max_features),
        "isolation_forest_standardize": isolation_forest_standardize,
    }


def _render_reporting_settings(defaults: dict, key_prefix: str) -> dict:
    """Reporting opt-in — shown identically at create time and in 'Adjust
    settings before training'. Off by default; nothing else in this
    function is required when it's off (see core/reporting/reporter.py's
    should_generate_report(), the single point that actually acts on
    these settings).
    """
    reporting_enabled = st.checkbox(
        "Enable AI-generated reports",
        value=bool(defaults["reporting_enabled"]),
        key=f"{key_prefix}_reporting_enabled",
        help=(
            "Optional layer on top of detection: after inspecting an image, retrieve relevant "
            "knowledge-base documentation and machine-context anomalies and generate a maintenance "
            "report. Detection itself is completely unaffected either way — this only adds a report. "
            "Knowledge documents and machine-parameter definitions (below, once enabled) are managed "
            "separately under 'Reporting: knowledge base & machine parameters' further down this page."
        ),
    )
    reporting_condition = defaults["reporting_condition"]
    reporting_classes_json = defaults["reporting_classes"]
    if reporting_enabled:
        condition_options = list(VALID_REPORTING_CONDITIONS)
        reporting_condition = st.selectbox(
            "Generate a report when...",
            condition_options,
            index=condition_options.index(defaults["reporting_condition"]),
            key=f"{key_prefix}_reporting_condition",
            help=(
                "'on_failed' (default): only when the verdict is failed. 'always': every inspection. "
                "'on_classes': only when the detected defect class is in the list below (not every "
                "method exposes one — see reporter.py). 'never': reporting stays enabled but no report "
                "is generated automatically, kept distinct from simply disabling reporting above."
            ),
        )
        if reporting_condition == "on_classes":
            existing_classes = ", ".join(json.loads(defaults["reporting_classes"] or "[]"))
            classes_text = st.text_input(
                "Classes that trigger a report (comma-separated)",
                value=existing_classes,
                key=f"{key_prefix}_reporting_classes",
            )
            reporting_classes_json = json.dumps([c.strip() for c in classes_text.split(",") if c.strip()])

    return {
        "reporting_enabled": reporting_enabled,
        "reporting_condition": reporting_condition,
        "reporting_classes": reporting_classes_json,
    }


# Analysis method lives outside any form so the rest of the page can react
# immediately to the choice (YOLO's creation flow needs several independent
# upload/draw steps, which can't be batched into one st.form submit anyway).
model_type_label = st.radio("Analysis method", list(MODEL_TYPE_OPTIONS.keys()))
model_type = MODEL_TYPE_OPTIONS[model_type_label]
failed_required = registry_factory.requires_failed_examples(model_type)
st.caption(MODEL_TYPE_DESCRIPTIONS[model_type])

if model_type == "yolo":
    st.subheader("Create a YOLO component")
    with st.form("create_yolo_component"):
        yolo_display_name = st.text_input("Component display name", placeholder="e.g. PCB Solder Joints")
        class_names_text = st.text_area(
            "Classes (one per line)",
            value="defect",
            help=(
                "What YOLO looks for. One class is enough for most defect-detection setups. "
                "If you use annotation path 1 (pre-made YOLO labels), the class IDs in your "
                "label files must match the order of this list."
            ),
        )

        with st.expander("Advanced settings"):
            yolo_advanced = _render_advanced_settings(SETTINGS_DEFAULTS, key_prefix="create_yolo")

        with st.expander("Reporting (optional)"):
            yolo_reporting = _render_reporting_settings(SETTINGS_DEFAULTS, key_prefix="create_yolo")

        ycol1, ycol2 = st.columns(2)
        yolo_image_size = ycol1.number_input(
            "Image size", min_value=32, step=32, value=SETTINGS_DEFAULTS["image_size"]
        )
        yolo_epochs = ycol2.number_input("Epochs", min_value=1, value=SETTINGS_DEFAULTS["epochs"])
        yolo_batch_size = ycol1.number_input(
            "Batch size", min_value=1, value=SETTINGS_DEFAULTS["batch_size"]
        )

        yolo_submitted = st.form_submit_button("Create component")

    if yolo_submitted:
        class_names = [c.strip() for c in class_names_text.splitlines() if c.strip()]
        if not yolo_display_name.strip():
            st.error("Please provide a display name.")
        elif not class_names:
            st.error("Please provide at least one class name.")
        else:
            yolo_component = onboard.create_yolo_component(
                yolo_display_name,
                class_names=class_names,
                image_size=int(yolo_image_size),
                epochs=int(yolo_epochs),
                batch_size=int(yolo_batch_size),
                registry=registry,
                **yolo_advanced,
                **yolo_reporting,
            )
            st.session_state["last_created_component"] = yolo_component.name
            st.session_state["yolo_annotate_component"] = yolo_component.name
            # A selectbox with an explicit key ignores `index=` on every rerun
            # after its first — its own prior selection in session_state wins.
            # Clearing it here is what lets the newly created component
            # actually become the new default instead of a stale prior pick.
            st.session_state.pop("yolo_annotate_select", None)
            st.session_state.pop("train_component_select", None)
            st.success(f"Created component '{yolo_component.display_name}' ({yolo_component.name}).")

    st.divider()
    st.subheader("Add annotations")

    yolo_components = [c for c in registry.list_all() if c.model_type == "yolo"]
    if not yolo_components:
        st.info("No YOLO components yet. Create one above first.")
    else:
        yolo_names = [c.name for c in yolo_components]
        default_annotate = st.session_state.get("yolo_annotate_component")
        default_annotate_idx = yolo_names.index(default_annotate) if default_annotate in yolo_names else 0
        yolo_target = st.selectbox(
            "Component to annotate",
            yolo_components,
            index=default_annotate_idx,
            format_func=lambda c: f"{c.display_name} ({c.status})",
            key="yolo_annotate_select",
        )
        st.session_state["yolo_annotate_component"] = yolo_target.name

        pool = onboard.yolo_annotation_summary(yolo_target.name)
        st.caption(
            f"Current pool: {pool['image_count']} images "
            f"({pool['positive_count']} with boxes, {pool['negative_count']} negative)."
        )
        class_names = onboard.get_yolo_classes(yolo_target.name)

        with st.expander(f"Manage classes ({len(class_names)} defined)"):
            st.caption(
                "Existing class indices are permanent: every saved annotation stores a bare "
                "class number, not a name, so reordering or removing a class here would silently "
                "repoint every label already saved against it. New classes can only be added at "
                "the end of the list."
            )
            if class_names:
                st.table({"Index": list(range(len(class_names))), "Class name": class_names})
            new_classes_text = st.text_area(
                "New class name(s), one per line",
                value="",
                key=f"new_classes_{yolo_target.name}",
            )
            if st.button("Add class(es)", key=f"add_classes_{yolo_target.name}"):
                new_names = [line.strip() for line in new_classes_text.splitlines() if line.strip()]
                if not new_names:
                    st.warning("Enter at least one class name.")
                else:
                    try:
                        updated = onboard.add_yolo_classes(yolo_target.name, new_names)
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        st.success(f"Added {len(new_names)} class(es). Full list: {', '.join(updated)}")
                        st.rerun()

        annotation_method = st.radio(
            "How will you provide annotations for this batch?",
            [
                "Already in YOLO format (images + .txt labels)",
                "Convert from segmentation masks (MVTec-style)",
                "Draw manually in the app",
            ],
            key=f"yolo_method_{yolo_target.name}",
        )

        # --- Path 1: pre-made YOLO-format labels -----------------------------
        if annotation_method.startswith("Already"):
            st.caption(
                "Upload images and their matching .txt label files (matched by filename, e.g. "
                "img001.png + img001.txt). An image with no matching label is saved as a "
                "negative example (no objects)."
            )
            p1_images = st.file_uploader(
                "Images", type=UPLOAD_TYPES, accept_multiple_files=True, key=f"p1_images_{yolo_target.name}"
            )
            p1_labels = st.file_uploader(
                "Label files (.txt)",
                type=["txt"],
                accept_multiple_files=True,
                key=f"p1_labels_{yolo_target.name}",
            )
            if st.button("Save these annotations", key=f"p1_save_{yolo_target.name}"):
                if not p1_images:
                    st.error("Please upload at least one image.")
                else:
                    labels_by_stem = {Path(f.name).stem: f for f in (p1_labels or [])}
                    saved = 0
                    for img_file in p1_images:
                        stem = Path(img_file.name).stem
                        label_file = labels_by_stem.get(stem)
                        label_text = label_file.getvalue().decode("utf-8") if label_file else ""
                        onboard.add_yolo_labeled_pair(
                            yolo_target.name, img_file.name, img_file.getvalue(), label_text
                        )
                        saved += 1
                    st.success(f"Saved {saved} image(s).")
                    st.rerun()

        # --- Path 2: mask-to-box conversion -----------------------------------
        elif annotation_method.startswith("Convert"):
            st.caption(
                "Upload images and their matching segmentation masks (matched by filename). "
                "Non-zero mask pixels are treated as defect regions; each disjoint region becomes "
                "its own box. Review the preview before saving — this is where a mis-scaled or "
                "mis-matched mask would show up."
            )
            if len(class_names) > 1:
                st.caption(
                    f"This component has {len(class_names)} classes; mask conversion always "
                    f"labels detected regions as class 0 ('{class_names[0]}')."
                )
            p2_images = st.file_uploader(
                "Images", type=UPLOAD_TYPES, accept_multiple_files=True, key=f"p2_images_{yolo_target.name}"
            )
            p2_masks = st.file_uploader(
                "Masks",
                type=UPLOAD_TYPES,
                accept_multiple_files=True,
                key=f"p2_masks_{yolo_target.name}",
            )

            if p2_images and p2_masks:
                masks_by_stem = {Path(f.name).stem: f for f in p2_masks}
                conversions = []
                for img_file in p2_images:
                    mask_file = masks_by_stem.get(Path(img_file.name).stem)
                    if mask_file is None:
                        continue
                    image_bytes = img_file.getvalue()
                    boxes = onboard.convert_mask_to_boxes(mask_file.getvalue())
                    conversions.append((img_file.name, image_bytes, boxes))

                if not conversions:
                    st.warning("No images and masks with matching filenames were found.")
                else:
                    st.caption(f"Preview ({len(conversions)} matched pairs):")
                    preview_cols = st.columns(3)
                    for i, (filename, image_bytes, boxes) in enumerate(conversions):
                        pil_image = Image.open(io.BytesIO(image_bytes))
                        annotated = yolo_annotation.render_boxes_on_image(pil_image, boxes, class_names)
                        with preview_cols[i % 3]:
                            st.image(annotated, caption=f"{filename} ({len(boxes)} box(es))", width="stretch")

                    if st.button("Save these annotations", key=f"p2_save_{yolo_target.name}"):
                        for filename, image_bytes, boxes in conversions:
                            onboard.add_yolo_annotation(yolo_target.name, filename, image_bytes, boxes)
                        st.success(f"Saved {len(conversions)} image(s).")
                        st.rerun()

        # --- Path 3: manual annotation -----------------------------------------
        else:
            queue_key = f"yolo_draw_queue_{yolo_target.name}"
            index_key = f"yolo_draw_index_{yolo_target.name}"

            p3_uploads = st.file_uploader(
                "Raw images to annotate",
                type=UPLOAD_TYPES,
                accept_multiple_files=True,
                key=f"p3_uploads_{yolo_target.name}",
            )
            if p3_uploads and st.button("Start annotating this batch", key=f"p3_start_{yolo_target.name}"):
                st.session_state[queue_key] = [(f.name, f.getvalue()) for f in p3_uploads]
                st.session_state[index_key] = 0
                st.rerun()

            queue = st.session_state.get(queue_key, [])
            index = st.session_state.get(index_key, 0)

            if not queue:
                st.info("Upload images above and click 'Start annotating this batch' to begin.")
            elif index >= len(queue):
                st.success(f"All {len(queue)} images in this batch are annotated.")
                if st.button("Clear batch", key=f"p3_clear_{yolo_target.name}"):
                    del st.session_state[queue_key]
                    del st.session_state[index_key]
                    st.rerun()
            else:
                filename, image_bytes = queue[index]
                st.caption(f"Image {index + 1} of {len(queue)}: {filename}")

                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                boxes = render_yolo_box_canvas(
                    pil_image, class_names, key_prefix=f"{yolo_target.name}_{filename}_{index}"
                )

                bcol1, bcol2 = st.columns(2)
                if bcol1.button(
                    "Save annotations & next", key=f"p3_save_{yolo_target.name}_{index}", disabled=not boxes
                ):
                    onboard.add_yolo_annotation(yolo_target.name, filename, image_bytes, boxes)
                    st.session_state[index_key] += 1
                    st.rerun()
                if bcol2.button("Skip (no defect) & next", key=f"p3_skip_{yolo_target.name}_{index}"):
                    onboard.add_yolo_annotation(yolo_target.name, filename, image_bytes, [])
                    st.session_state[index_key] += 1
                    st.rerun()

            st.caption(
                "Note: combining automatic mask conversion with manual correction in this same "
                "view (pre-loading detected boxes so you can adjust/remove/add) is a natural "
                "next step, kept separate for now to avoid overcomplicating the first version."
            )

elif model_type == "coco_detector":
    st.subheader("Create an object & face cascade component")
    st.caption(
        "Frozen, pretrained COCO-YOLO — no dataset, no annotation. Just a name; it's ready to "
        "use the moment it's created."
    )
    with st.form("create_coco_detector_component"):
        coco_display_name = st.text_input("Component display name", placeholder="e.g. Entrance Camera")
        coco_submitted = st.form_submit_button("Create component")

    if coco_submitted:
        if not coco_display_name.strip():
            st.error("Please provide a display name.")
        else:
            coco_component = onboard.create_component(
                coco_display_name, model_type="coco_detector", registry=registry
            )
            # No data to wait for — "training" this model_type is a fast
            # no-op (see core/detection/yolo_coco/trainer.py) that just
            # flips status to 'ready', so do it immediately rather than
            # making the operator find a separate Train button for a
            # component that has nothing to train.
            onboard.train_component(coco_component.name, registry=registry)
            st.session_state["last_created_component"] = coco_component.name
            st.session_state.pop("train_component_select", None)  # see comment on the YOLO branch above
            st.success(
                f"Created and readied '{coco_component.display_name}' ({coco_component.name}). "
                "Configure it in the 'Cascade: object & face recognition' section below."
            )

elif model_type in AVAILABLE_MODEL_TYPES:
    with st.form("create_component"):
        display_name = st.text_input("Component display name", placeholder="e.g. Bracket Type A")

        with st.expander("Advanced settings"):
            advanced = _render_advanced_settings(SETTINGS_DEFAULTS, key_prefix="create")

        with st.expander("Reporting (optional)"):
            reporting_settings = _render_reporting_settings(SETTINGS_DEFAULTS, key_prefix="create")

        col1, col2 = st.columns(2)
        image_size = col1.number_input(
            "Image size", min_value=32, step=16, value=SETTINGS_DEFAULTS["image_size"]
        )
        epochs = col2.number_input("Epochs", min_value=1, value=SETTINGS_DEFAULTS["epochs"])
        latent_dim = col1.number_input(
            "Latent dimension", min_value=2, value=SETTINGS_DEFAULTS["latent_dim"]
        )
        batch_size = col2.number_input(
            "Batch size", min_value=1, value=SETTINGS_DEFAULTS["batch_size"]
        )

        approved_files = st.file_uploader(
            "Approved (normal) files", type=UPLOAD_TYPES, accept_multiple_files=True
        )
        failed_label = "Failed (anomalous) files" + (
            " — required for this method" if failed_required else " — optional, used to validate the threshold"
        )
        failed_files = st.file_uploader(failed_label, type=UPLOAD_TYPES, accept_multiple_files=True)

        submitted = st.form_submit_button("Create component")

    if submitted:
        if not display_name.strip():
            st.error("Please provide a display name.")
        elif not approved_files:
            st.error("Please upload at least one approved file.")
        elif failed_required and not failed_files:
            st.error(f"'{model_type_label}' requires at least one failed file to train on.")
        else:
            component = onboard.create_component(
                display_name,
                model_type=model_type,
                image_size=int(image_size),
                epochs=int(epochs),
                latent_dim=int(latent_dim),
                batch_size=int(batch_size),
                registry=registry,
                **advanced,
                **reporting_settings,
            )
            onboard.add_training_images(
                component.name,
                approved=[(f.name, f.getvalue()) for f in approved_files],
                failed=[(f.name, f.getvalue()) for f in (failed_files or [])],
            )
            st.session_state["last_created_component"] = component.name
            st.session_state.pop("train_component_select", None)  # see comment on the YOLO branch above
            st.success(f"Created component '{component.display_name}' ({component.name}).")

else:
    st.info(
        f"'{model_type_label}' isn't implemented yet — please choose Autoencoder, Classifier, "
        "YOLO, or PatchCore."
    )

st.divider()
st.subheader("Edit component")
st.caption(
    "Change settings that don't affect the trained model itself — display name, reporting, "
    "how long acknowledged inspections are kept — without retraining. Settings that DO affect "
    "the model (image size, epochs, per-method hyperparameters, ...) live under 'Train a "
    "component' below, since changing those only takes effect on the next training run anyway."
)

editable = registry.list_all()
if not editable:
    st.info("No components yet. Create one above first.")
else:
    to_edit = st.selectbox(
        "Component to edit",
        editable,
        format_func=lambda c: f"{c.display_name} ({c.status}, {c.model_type})"
        + (" [inactive]" if c.lifecycle_status == "inactive" else ""),
        key="edit_component_select",
    )
    edit_settings_snapshot = {key: getattr(to_edit, key) for key in SETTINGS_DEFAULTS}

    edit_display_name = st.text_input(
        "Display name", value=to_edit.display_name, key=f"edit_display_name_{to_edit.name}"
    )
    st.caption(
        f"Internal identifier stays '{to_edit.name}' regardless — this only changes the label "
        "shown in the app (folder names, indexed documentation, and inspection history all key "
        "off the internal identifier, untouched by a display-name edit)."
    )

    with st.expander("Reporting (optional)"):
        edit_reporting_only = _render_reporting_settings(edit_settings_snapshot, key_prefix=f"editonly_{to_edit.name}")

    edit_retention_days = st.number_input(
        "Keep archived inspections for (days)",
        min_value=0,
        value=int(to_edit.inspection_retention_days),
        key=f"edit_retention_{to_edit.name}",
        help=(
            "The THIRD lifecycle step only: archived -> deleted. How long an already-archived "
            "inspection's files are kept before the cleanup button below removes them. Has no "
            "effect on the first two steps — new -> acknowledged (kvittering) or acknowledged -> "
            "archived (arkivering, manual/bulk/auto) both happen on their own schedule regardless "
            "of this setting. 0 means keep archived inspections indefinitely (retention off), not "
            "delete immediately. Never applies to a human-verified correction not yet used in a "
            "retrain — see core/inspections/retention.py. Not run automatically — the button below "
            "is the only way this actually deletes anything."
        ),
    )
    edit_approved_handling = st.selectbox(
        "Approved verdicts",
        VALID_APPROVED_HANDLING_MODES,
        index=VALID_APPROVED_HANDLING_MODES.index(to_edit.approved_handling),
        key=f"edit_approved_handling_{to_edit.name}",
        format_func=lambda mode: {
            "keep_visible": "Keep visible — shown on the Inspection Station like any other record",
            "hide_from_default_view": "Hide from default view — always reachable via its filter (recommended)",
            "auto_acknowledge": "Auto-acknowledge — no manual review needed, still archived on the usual schedule",
        }[mode],
        help="Approved records are never deleted or made permanently unreachable regardless of "
        "this setting — it only controls default visibility and whether acknowledgement is manual.",
    )
    edit_verified_correction_policy = st.selectbox(
        "Verified corrections -> training data",
        VALID_VERIFIED_CORRECTION_POLICIES,
        index=VALID_VERIFIED_CORRECTION_POLICIES.index(to_edit.verified_correction_policy),
        key=f"edit_verified_correction_policy_{to_edit.name}",
        format_func=lambda mode: {
            "off": "Off — corrections stay verified-only, never copied into training/ automatically",
            "manual_review": "Manual review — a human picks which pending corrections go in, at the Train section below (recommended)",
            "automatic": "Automatic — every pending correction is copied into training/ with no review",
        }[mode],
        help="A single mis-annotated correction copied in unreviewed can quietly poison the "
        "training set — this controls how much human judgment sits between a verified correction "
        "and it actually becoming training data. 'Automatic' removes a review point the same way "
        "auto-acknowledge does for approved verdicts above; 'Manual review' is the quality gate; "
        "'Off' keeps training data completely stable regardless of how many corrections pile up.",
    )

    if st.button(
        f"Run retention cleanup for '{to_edit.display_name}' now",
        key=f"run_retention_{to_edit.name}",
        help="Permanently deletes archived inspections older than the retention setting above — "
        "the only action anywhere in this app that actually deletes an inspection's files and "
        "database row, rather than just moving or flagging it. A human-verified correction not "
        "yet incorporated into a retrain is always protected, however old it is.",
    ):
        result = retention.cleanup_archived_inspections(to_edit.name, registry=registry)
        if result.deleted:
            st.success(f"Deleted {result.deleted} archived inspection(s) past retention.")
        else:
            st.info("Nothing was old enough to delete.")
        if result.protected_pending_verified:
            st.warning(
                f"Skipped {result.protected_pending_verified} record(s) — verified corrections not "
                "yet incorporated into a retrain (see the Inspection Station for details)."
            )
        if result.errors:
            st.error(f"{len(result.errors)} record(s) could not be deleted: {result.errors[:5]}")

    if st.button("Save changes", key=f"save_edit_{to_edit.name}"):
        if not edit_display_name.strip():
            st.error("Display name cannot be empty.")
        else:
            if edit_display_name.strip() != to_edit.display_name:
                registry.rename_display_name(to_edit.name, edit_display_name.strip())
            registry.update_settings(
                to_edit.name,
                inspection_retention_days=int(edit_retention_days),
                approved_handling=edit_approved_handling,
                verified_correction_policy=edit_verified_correction_policy,
                **edit_reporting_only,
            )
            st.success(f"Saved changes to '{edit_display_name.strip()}'. No retraining needed.")
            st.rerun()

    st.divider()
    st.markdown("**Component status**")
    st.caption(
        "Deactivate/reactivate and delete (to trash) are both reversible, zero-impact actions — "
        "nothing about this component's models, training data, inspection history, or knowledge "
        "base is ever touched by either one; they only control whether it shows up for active use "
        "(the watcher, RAG reindexing, the Inspect page) or in this management view. Only "
        "'Delete permanently' in the Trash section below is irreversible."
    )
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        if to_edit.lifecycle_status == "active":
            if st.button("⏸ Deactivate", key=f"deactivate_{to_edit.name}", help="Pauses this component — excluded from the watcher, RAG reindexing, and the Inspect page's picker until reactivated. Nothing is deleted."):
                registry.deactivate(to_edit.name)
                st.success(f"Deactivated '{to_edit.display_name}'.")
                st.rerun()
        else:
            if st.button("▶ Reactivate", key=f"reactivate_{to_edit.name}", help="Resumes normal active use — a simple flag flip back, nothing to restore since nothing was removed."):
                registry.reactivate(to_edit.name)
                st.success(f"Reactivated '{to_edit.display_name}'.")
                st.rerun()
    with status_col2:
        if st.button(
            "🗑 Delete (move to trash)",
            key=f"soft_delete_{to_edit.name}",
            help="Hides this component from active use AND from this management view, but keeps "
            "every bit of its data untouched — reachable and fully restorable from the Trash "
            "section below at any time.",
        ):
            registry.soft_delete(to_edit.name)
            st.success(f"Moved '{to_edit.display_name}' to the trash — restore it below anytime.")
            st.rerun()

st.divider()
st.subheader("Cascade: object & face recognition")
st.caption(
    "Configure and run the object-detection + face-recognition cascade for an 'Object & face "
    "cascade (COCO detector)' component — register consenting individuals, set what happens "
    "when someone is recognized (or not), and try it on a test image. This calls the exact same "
    "run_cascade() a future live video/Kafka frame source would call per frame — only where the "
    "frame comes from differs."
)

cascade_components = [c for c in registry.list_all() if c.model_type == "coco_detector"]
if not cascade_components:
    st.info(
        "No cascade components yet. Create one above — pick 'Object & face cascade (COCO "
        "detector)' as the Analysis method."
    )
else:
    cascade_names = [c.name for c in cascade_components]
    default_cascade_name = st.session_state.get("last_created_component")
    default_cascade_idx = cascade_names.index(default_cascade_name) if default_cascade_name in cascade_names else 0
    to_cascade = st.selectbox(
        "Cascade component",
        cascade_components,
        index=default_cascade_idx,
        format_func=lambda c: f"{c.display_name} ({c.status})",
        key="cascade_component_select",
    )

    st.markdown("#### Known individuals")
    st.caption(
        "Only people registered here, with their explicit consent, are ever identified by name — "
        "everyone else is always 'unknown', by design. Multiple photos per person (ideally under "
        "varying angles/lighting/expression) make matching more robust and let the threshold "
        "below be calibrated against real observed distances instead of guessed."
    )

    known_individuals = face_store.list_known_individuals()
    if known_individuals:
        for person in known_individuals:
            with st.expander(f"{person.name} — {person.embedding_count} photo(s)"):
                st.caption(
                    f"registered {person.created_at}" + (f" — {person.notes}" if person.notes else "")
                )

                person_embeddings = face_store.list_embeddings_for(person.identity_key)
                if person_embeddings:
                    for photo_index, embedding_record in enumerate(person_embeddings, start=1):
                        ecol1, ecol2 = st.columns([3, 1])
                        ecol1.caption(f"Photo #{photo_index} — added {embedding_record.created_at}")
                        if ecol2.button("Remove", key=f"remove_embedding_{embedding_record.id}"):
                            face_store.delete_face_embedding(embedding_record.id)
                            st.success(f"Removed photo #{photo_index} from '{person.name}'.")
                            st.rerun()
                else:
                    st.warning(
                        "No photos registered — this person can never be matched until at least "
                        "one is added below."
                    )

                st.markdown("**Add another photo**")
                st.caption(
                    "A photo taken under different conditions than the ones already registered "
                    "(angle, lighting, expression) helps most — not required, just more useful "
                    "than a near-duplicate of an existing one."
                )
                add_photo_upload = st.file_uploader(
                    "Photo", type=UPLOAD_TYPES, key=f"add_photo_{person.identity_key}"
                )
                if add_photo_upload is not None:
                    add_photo_image = Image.open(io.BytesIO(add_photo_upload.getvalue())).convert("RGB")
                    with st.spinner("Detecting faces..."):
                        add_photo_faces = detect_all_faces(add_photo_image)

                    if not add_photo_faces:
                        st.warning("No face detected in this photo — try a different one.")
                    else:
                        add_photo_preview_detections = [
                            {"class": f"Face {i + 1}", "confidence": f.confidence, "box": list(f.box)}
                            for i, f in enumerate(add_photo_faces)
                        ]
                        st.image(
                            yolo_annotation.render_boxes_on_image(add_photo_image, add_photo_preview_detections),
                            caption=f"{len(add_photo_faces)} face(s) detected",
                            width="stretch",
                        )
                        if len(add_photo_faces) == 1:
                            add_photo_chosen_index = 0
                        else:
                            add_photo_labels = [
                                f"Face {i + 1} (confidence {f.confidence:.2f})"
                                for i, f in enumerate(add_photo_faces)
                            ]
                            add_photo_chosen_label = st.radio(
                                "Which face is this person?",
                                add_photo_labels,
                                key=f"add_photo_choice_{person.identity_key}",
                            )
                            add_photo_chosen_index = add_photo_labels.index(add_photo_chosen_label)

                        if st.button("Add this photo", key=f"add_photo_submit_{person.identity_key}"):
                            face_store.add_face_embedding(
                                person.identity_key, add_photo_faces[add_photo_chosen_index].embedding
                            )
                            st.success(f"Added a new photo to '{person.name}'.")
                            st.rerun()

                st.divider()
                if st.button(
                    f"Remove '{person.name}' entirely (all {person.embedding_count} photo(s))",
                    key=f"remove_person_{person.identity_key}",
                ):
                    face_store.delete_known_individual(person.identity_key)
                    st.success(f"Removed '{person.name}' — every one of their photos has been permanently deleted.")
                    st.rerun()
    else:
        st.caption("No individuals registered yet.")

    with st.expander("Register a new individual", expanded=(len(known_individuals) == 0)):
        st.caption(
            "This registers the person with their FIRST photo. Add more photos afterward from "
            "their entry above for more robust matching."
        )
        reg_photo = st.file_uploader("Photo", type=UPLOAD_TYPES, key=f"face_reg_photo_{to_cascade.name}")
        if reg_photo is None:
            st.caption("Upload a photo to detect a face.")
        else:
            reg_image = Image.open(io.BytesIO(reg_photo.getvalue())).convert("RGB")
            with st.spinner("Detecting faces..."):
                detected_faces = detect_all_faces(reg_image)

            if not detected_faces:
                st.warning(
                    "No face detected in this photo — try a clearer, more front-facing photo, "
                    "or upload a different one."
                )
            else:
                preview_detections = [
                    {"class": f"Face {i + 1}", "confidence": f.confidence, "box": list(f.box)}
                    for i, f in enumerate(detected_faces)
                ]
                preview_image = yolo_annotation.render_boxes_on_image(reg_image, preview_detections)
                st.image(preview_image, caption=f"{len(detected_faces)} face(s) detected", width="stretch")

                if len(detected_faces) == 1:
                    chosen_index = 0
                    st.caption("One face detected — this is the one that will be registered.")
                else:
                    st.warning(f"{len(detected_faces)} faces detected — pick which one to register.")
                    face_choice_labels = [
                        f"Face {i + 1} (confidence {f.confidence:.2f})" for i, f in enumerate(detected_faces)
                    ]
                    chosen_label = st.radio(
                        "Which face is the person you're registering?",
                        face_choice_labels,
                        key=f"face_reg_choice_{to_cascade.name}",
                    )
                    chosen_index = face_choice_labels.index(chosen_label)

                chosen_face = detected_faces[chosen_index]
                x1, y1, x2, y2 = chosen_face.box
                cropped_face = reg_image.crop((x1, y1, x2, y2))
                crop_col, form_col = st.columns([1, 2])
                crop_col.image(cropped_face, caption="Selected face", width=180)

                with form_col:
                    reg_name = st.text_input("Person's name", key=f"face_reg_name_{to_cascade.name}")
                    reg_consent = st.checkbox(
                        "This person has given their explicit consent for their face data to be "
                        "stored and used for recognition.",
                        value=False,
                        key=f"face_reg_consent_{to_cascade.name}",
                        help="Required — registration stays blocked until this is actively "
                        "checked. Nobody is ever added to the recognizable-individuals database "
                        "without this box being deliberately ticked; it is never pre-checked or "
                        "implied.",
                    )
                    if not reg_consent:
                        st.caption("⚠️ Consent must be given before registration is possible.")
                    register_clicked = st.button(
                        "Register this individual",
                        key=f"face_reg_submit_{to_cascade.name}",
                        disabled=not (reg_consent and reg_name.strip()),
                    )

                if register_clicked:
                    face_store.add_known_individual(reg_name.strip(), chosen_face.embedding, consented=True)
                    st.success(f"Registered '{reg_name.strip()}' — they will now be recognized by the cascade.")
                    st.rerun()

    with st.expander("Threshold calibration"):
        st.caption(
            "How well do the registered individuals' photos actually separate, at the SAME "
            "embedding distance the matcher itself uses? Calibrating against an observed "
            "distribution instead of guessing — the same principle the anomaly-detection "
            "methods' evaluation reports (Training history, further up) already apply to their "
            "own threshold. Needs at least one person with 2+ photos (to measure same-person "
            "spread) and at least two people (to measure different-person distance)."
        )
        calibration = face_store.calibration_stats()
        if not calibration.intra_person_distances or not calibration.inter_person_distances:
            st.caption(
                "Not enough data yet — register at least one individual with 2+ photos, and at "
                "least two individuals total, to see calibration here."
            )
        else:
            cal_col1, cal_col2, cal_col3 = st.columns(3)
            cal_col1.metric("Max same-person distance", f"{max(calibration.intra_person_distances):.3f}")
            cal_col2.metric("Min different-person distance", f"{min(calibration.inter_person_distances):.3f}")
            cal_col3.metric("Current threshold", f"{DEFAULT_FACE_MATCH_DISTANCE_THRESHOLD:.3f}")

            if calibration.separable:
                st.success(
                    f"Separable — suggested threshold ≈ {calibration.suggested_threshold:.3f} "
                    f"(sits between the two distributions below)."
                )
            else:
                st.warning(
                    "These registered individuals' photos do NOT cleanly separate — the worst "
                    "same-person distance is larger than the closest different-person distance. "
                    "No single threshold perfectly distinguishes them with the current photos; "
                    "consider adding more/better photos for the people involved."
                )

            calibration_chart_data = pd.DataFrame(
                [{"distance": d, "type": "same person"} for d in calibration.intra_person_distances]
                + [{"distance": d, "type": "different people"} for d in calibration.inter_person_distances]
            )
            distance_histogram = (
                alt.Chart(calibration_chart_data)
                .mark_bar(opacity=0.6)
                .encode(
                    x=alt.X("distance:Q", bin=alt.Bin(maxbins=30), title="Embedding L2 distance"),
                    y=alt.Y("count()", title="Count", stack=None),
                    color=alt.Color("type:N", title=""),
                )
            )
            threshold_rule = (
                alt.Chart(pd.DataFrame({"threshold": [DEFAULT_FACE_MATCH_DISTANCE_THRESHOLD]}))
                .mark_rule(color="black", strokeDash=[4, 4])
                .encode(x="threshold:Q")
            )
            st.altair_chart(distance_histogram + threshold_rule, width="stretch")

    with st.expander("Category → specialist activation"):
        st.caption(
            "Which coarse categories run a specialist (and which one) for this component. A "
            "category with no specialist activated is still detected and reported in 'Run the "
            "cascade' below — its real object class and coarse category stay fully visible — "
            "just without further identification. Default: only 'human' activates anything "
            "(face recognition). A future car-model specialist could be turned on for 'vehicle' "
            "here with no code change."
        )
        current_category_specialists = specialist_registry.parse_category_specialists(
            to_cascade.cascade_category_specialists
        )
        specialist_choice_options = [NO_SPECIALIST_OPTION, *specialist_registry.available_specialist_names()]

        new_category_specialists: dict[str, str] = {}
        for category in ALL_CASCADE_CATEGORIES:
            current_choice = current_category_specialists.get(category, "")
            default_index = (
                specialist_choice_options.index(current_choice)
                if current_choice in specialist_choice_options
                else 0
            )
            chosen_specialist = st.selectbox(
                f"'{category}' activates",
                specialist_choice_options,
                index=default_index,
                key=f"category_specialist_{to_cascade.name}_{category}",
            )
            if chosen_specialist != NO_SPECIALIST_OPTION:
                new_category_specialists[category] = chosen_specialist

        if st.button("Save category → specialist configuration", key=f"save_category_specialists_{to_cascade.name}"):
            registry.update_settings(
                to_cascade.name,
                cascade_category_specialists=specialist_registry.serialize_category_specialists(
                    new_category_specialists
                ),
            )
            st.success("Saved — takes effect on the next cascade run.")
            st.rerun()

    st.markdown("#### Reaction policies")
    st.caption(
        "What happens when the face specialist identifies someone (or doesn't). 'unknown' is a "
        "first-class case with its own policy, set below — not a silent fallback."
    )

    policy_targets: dict[str, str] = {"unknown": "Unknown (no match)"}
    for person in known_individuals:
        policy_targets[person.identity_key] = person.name

    existing_policies = {p.identity_key: p for p in policy_store.list_policies("face")}
    if "unknown" not in existing_policies:
        st.warning(
            "No policy configured yet for 'unknown' — until you set one below, an unrecognized "
            "person is only logged silently, with no alert. Configure it explicitly below instead "
            "of relying on that default."
        )

    if existing_policies:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "identity": policy_targets.get(key, key),
                        "label": p.label,
                        "message": p.message,
                        "actions": ", ".join(p.actions),
                        "priority": p.priority,
                    }
                    for key, p in existing_policies.items()
                ]
            ),
            width="stretch",
        )
    else:
        st.caption("No policies configured yet.")

    with st.expander("Set a policy", expanded=("unknown" not in existing_policies)):
        policy_target_key = st.selectbox(
            "Configure policy for",
            list(policy_targets.keys()),
            format_func=lambda k: policy_targets[k],
            key=f"policy_target_{to_cascade.name}",
        )
        current_policy = existing_policies.get(policy_target_key)
        is_unknown = policy_target_key == "unknown"

        policy_label = st.text_input(
            "Label",
            value=current_policy.label if current_policy else ("unknown" if is_unknown else "approved person"),
            key=f"policy_label_{to_cascade.name}_{policy_target_key}",
        )
        policy_message = st.text_input(
            "Message",
            value=current_policy.message
            if current_policy
            else ("Unrecognized individual detected." if is_unknown else "Welcome back."),
            key=f"policy_message_{to_cascade.name}_{policy_target_key}",
        )
        policy_actions = st.multiselect(
            "Actions",
            VALID_ACTIONS,
            default=list(current_policy.actions)
            if current_policy
            else (["log", "alert", "save_frame"] if is_unknown else ["log"]),
            key=f"policy_actions_{to_cascade.name}_{policy_target_key}",
        )
        policy_priority = st.selectbox(
            "Priority",
            VALID_PRIORITIES,
            index=VALID_PRIORITIES.index(
                current_policy.priority if current_policy else ("high" if is_unknown else DEFAULT_PRIORITY)
            ),
            key=f"policy_priority_{to_cascade.name}_{policy_target_key}",
        )
        if st.button("Save policy", key=f"policy_save_{to_cascade.name}_{policy_target_key}"):
            policy_store.upsert_policy(
                "face", policy_target_key,
                label=policy_label, message=policy_message, actions=policy_actions, priority=policy_priority,
            )
            st.success(f"Saved policy for '{policy_targets[policy_target_key]}'.")
            st.rerun()

    st.markdown("#### Run the cascade")
    st.caption(
        "Upload a test image to run the full cascade: object detection → face recognition (for "
        "people) → reaction policy."
    )
    if to_cascade.status != "ready":
        st.warning(f"'{to_cascade.display_name}' isn't ready yet.")
    else:
        cascade_test_upload = st.file_uploader(
            "Test image", type=UPLOAD_TYPES, key=f"cascade_run_upload_{to_cascade.name}"
        )
        cascade_result_key = f"cascade_last_result_{to_cascade.name}"
        if cascade_test_upload is not None and st.button("Run cascade", key=f"cascade_run_button_{to_cascade.name}"):
            cascade_image_bytes = cascade_test_upload.getvalue()
            with st.spinner("Running cascade..."):
                cascade_result = cascade_pipeline.run_cascade(cascade_image_bytes, to_cascade.name, registry=registry)
            source_image = Image.open(io.BytesIO(cascade_image_bytes)).convert("RGB")
            drawable = [
                {"class": obj.label, "confidence": obj.confidence, "box": list(obj.box)}
                for obj in cascade_result.objects
                if obj.box is not None
            ]
            annotated_image = (
                yolo_annotation.render_boxes_on_image(source_image, drawable) if drawable else source_image
            )
            st.session_state[cascade_result_key] = (annotated_image, cascade_result)

        if cascade_result_key in st.session_state:
            annotated_image, cascade_result = st.session_state[cascade_result_key]
            st.image(annotated_image, width="stretch")
            if not cascade_result.objects:
                st.info("No objects detected in this image.")
            for obj in cascade_result.objects:
                with st.container(border=True):
                    result_col1, result_col2 = st.columns([1, 2])
                    result_col1.markdown(f"**{obj.label}**")
                    result_col1.caption(f"category: {obj.category} · confidence: {obj.confidence:.2f}")
                    if obj.specialist_result is None:
                        result_col2.caption(
                            "No specialist activated for this category on this component — "
                            "detected and reported only (see 'Category → specialist activation' "
                            "above to turn one on)."
                        )
                    else:
                        sr = obj.specialist_result
                        if sr.matched:
                            result_col2.success(f"Recognized: **{sr.identity_label}**")
                        else:
                            result_col2.warning(f"Unknown person ({sr.details.get('reason', 'no match')})")
                        if obj.policy_result is not None:
                            pr = obj.policy_result
                            result_col2.markdown(f"**{pr.policy.label}** — _{pr.policy.message}_")
                            result_col2.caption(
                                f"actions triggered: {', '.join(pr.executed_actions) or '(none)'} · "
                                f"priority: {pr.policy.priority}"
                            )

st.divider()
st.subheader("Trash")
st.caption(
    "Soft-deleted components — hidden from active use and from the sections above, but every "
    "bit of their data (models, training data, inspection history, knowledge base, verified "
    "corrections) is untouched and fully restorable. Only 'Delete permanently' below actually "
    "destroys anything, and only after you confirm exactly what that includes."
)

trashed = registry.list_deleted()
if not trashed:
    st.caption("Nothing in the trash.")
else:
    if st.button(
        "Run cleanup now (permanently delete everything past the retention window)",
        key="run_trash_cleanup",
        help=f"Permanently deletes every trashed component older than "
        f"{DEFAULT_COMPONENT_DELETION_RETENTION_DAYS} days. A component can also be deleted "
        "permanently right away via its own button below, regardless of this window.",
    ):
        cleanup_results = component_deletion.cleanup_expired_soft_deleted_components(registry=registry)
        if cleanup_results:
            st.success(f"Permanently deleted {len(cleanup_results)} component(s) past the retention window.")
            for name, result in cleanup_results.items():
                if result.errors:
                    st.error(f"'{name}': completed with errors: {result.errors}")
        else:
            st.caption("Nothing in the trash is past the retention window yet.")
        st.rerun()

    for trashed_component in trashed:
        with st.expander(f"{trashed_component.display_name} (deleted {trashed_component.deleted_at})"):
            tcol1, tcol2 = st.columns(2)
            if tcol1.button("↩ Restore", key=f"restore_component_{trashed_component.name}"):
                registry.restore(trashed_component.name)
                st.success(f"Restored '{trashed_component.display_name}'.")
                st.rerun()

            confirm_key = f"confirm_permanent_delete_{trashed_component.name}"
            if not st.session_state.get(confirm_key, False):
                if tcol2.button("🗑 Delete permanently...", key=f"start_permanent_delete_{trashed_component.name}"):
                    st.session_state[confirm_key] = True
                    st.rerun()
            else:
                impact = component_deletion.summarize_deletion_impact(trashed_component.name, registry=registry)
                st.warning(
                    f"**This cannot be undone.** Permanently deleting '{trashed_component.display_name}' removes:\n\n"
                    f"- {impact.inspection_count} inspection(s)\n"
                    f"- **{impact.verified_correction_count} human-verified correction(s) — irreplaceable, cannot be regenerated**\n"
                    f"- {'an active trained model' if impact.has_active_model else 'no active model'}"
                    f" + {impact.model_backup_count} model backup(s)\n"
                    f"- {impact.training_file_count} training file(s)\n"
                    f"- {impact.training_run_count} recorded training run(s) (performance history)\n"
                    f"- {impact.knowledge_document_count} knowledge document(s), "
                    f"{impact.chromadb_chunk_count} indexed chunk(s)\n"
                    f"- {impact.machine_reading_count} machine reading(s)\n"
                    f"- {impact.cascade_stream_result_count} cascade stream result(s)\n\n"
                    "All of it — filesystem, ChromaDB, and database — with nothing left for a "
                    "future component with the same name to inherit."
                )
                ccol1, ccol2 = st.columns(2)
                if ccol1.button(
                    f"Yes, permanently delete '{trashed_component.display_name}'",
                    key=f"confirm_permanent_delete_btn_{trashed_component.name}",
                    type="primary",
                ):
                    result = component_deletion.permanently_delete_component(trashed_component.name, registry=registry)
                    st.session_state[confirm_key] = False
                    if result.errors:
                        st.error(f"Completed with errors: {result.errors}")
                    else:
                        st.success(f"'{trashed_component.display_name}' permanently deleted.")
                    st.rerun()
                if ccol2.button("Cancel", key=f"cancel_permanent_delete_{trashed_component.name}"):
                    st.session_state[confirm_key] = False
                    st.rerun()

st.divider()
st.subheader("Train a component")

trainable = registry.list_all()
if not trainable:
    st.info("No components yet. Create one above first.")
else:
    names = [c.name for c in trainable]
    last_created = st.session_state.get("last_created_component")
    default_index = names.index(last_created) if last_created in names else 0

    to_train = st.selectbox(
        "Component to train",
        trainable,
        index=default_index,
        format_func=lambda c: f"{c.display_name} ({c.status}, {c.model_type})",
        key="train_component_select",
    )
    st.caption(
        "Retraining a 'ready' component overwrites its model with a fresh training run — "
        "e.g. after adjusting settings below, or uploading more training files."
    )
    backup_before_training = st.checkbox(
        "Save current model as backup before training",
        value=True,
        key=f"backup_before_training_{to_train.name}",
        help="Copies the current model file to models/ with a timestamped name "
        "(<model>_bakYYMMDDHHMM.<ext>) before this run overwrites it, so it can be "
        "restored below if the new training run turns out worse.",
    )

    def _update_evaluation_report_inline(component_name: str, widget_key: str) -> None:
        registry.update_settings(component_name, generate_evaluation_report=st.session_state[widget_key])

    eval_report_key = f"generate_evaluation_report_{to_train.name}"
    st.checkbox(
        "Generate evaluation report after training",
        value=bool(to_train.generate_evaluation_report),
        key=eval_report_key,
        on_change=_update_evaluation_report_inline,
        args=(to_train.name, eval_report_key),
        help="Saves model_type-aware metrics (classification report / mAP / threshold precision-recall, "
        "depending on method) and plots to evaluation/<timestamp>/ next to the model — timestamped, never "
        "overwritten, so a before/after comparison across retrains is always possible. Turn off for a "
        "quick round of iteration where you don't need a full report each time. Changing this here "
        "applies immediately.",
    )
    selected_correction_ids: list[int] = []
    pending_corrections = onboard.list_pending_verified_corrections(to_train.name)

    st.markdown("**Verified corrections**")

    def _update_verified_policy_inline(component_name: str, widget_key: str) -> None:
        registry.update_settings(component_name, verified_correction_policy=st.session_state[widget_key])

    train_policy_key = f"train_verified_policy_{to_train.name}"
    st.selectbox(
        "Verified corrections -> training data",
        VALID_VERIFIED_CORRECTION_POLICIES,
        index=VALID_VERIFIED_CORRECTION_POLICIES.index(to_train.verified_correction_policy),
        key=train_policy_key,
        on_change=_update_verified_policy_inline,
        args=(to_train.name, train_policy_key),
        format_func=lambda mode: {
            "off": "Off — never copied into training/ automatically",
            "manual_review": "Manual review — pick which pending corrections go in below (recommended)",
            "automatic": "Automatic — every pending correction is copied in with no review",
        }[mode],
        help="Same setting as 'Verified corrections -> training data' in the Edit component "
        "section above — changing it here applies immediately, no separate save needed.",
    )

    if to_train.verified_correction_policy == "off":
        if pending_corrections:
            st.caption(
                f"{len(pending_corrections)} verified correction(s) are pending but this "
                "component's policy is 'off' — none will be copied into training/ automatically. "
                "Switch the policy above to 'manual_review' or 'automatic' to use them."
            )
    elif not pending_corrections:
        st.caption("No pending verified corrections right now.")
    elif to_train.verified_correction_policy == "automatic":
        st.caption(
            f"{len(pending_corrections)} pending verified correction(s) will be copied into "
            "training/ automatically before this run (policy: automatic — no review step)."
        )
        selected_correction_ids = [r.id for r in pending_corrections]
    else:  # manual_review — the actual quality gate: a human picks which ones go in
        with st.expander(f"Pending verified corrections ({len(pending_corrections)}) — choose which to incorporate", expanded=True):
            st.caption(
                "Unchecked corrections stay pending and are offered again next time — nothing is "
                "lost or force-included. This is what catches a mis-annotated correction before it "
                "ever reaches training/."
            )
            for record in pending_corrections:
                label = record.verified_label or {}
                kind = (
                    "✅ confirmed correct"
                    if record.verified_status == "verified_correct"
                    else f"🚩 {(record.verified_error_type or 'corrected').replace('_', ' ')}"
                )
                checked = st.checkbox(
                    f"#{record.id} — {kind} — verdict: {label.get('verdict', '?')}"
                    + (f", classes: {', '.join(label.get('defect_classes', []))}" if label.get("defect_classes") else "")
                    + f" — {record.created_at}",
                    value=True,
                    key=f"incorporate_correction_{to_train.name}_{record.id}",
                )
                if checked:
                    selected_correction_ids.append(record.id)

    with st.expander(f"Model backups & rollback ({to_train.display_name})"):
        backups = onboard.list_model_backups(to_train.name)
        if not to_train.model_path:
            st.caption("No model trained yet — nothing to back up or roll back.")
        elif not backups:
            st.caption("No backups yet. One is created automatically before a retrain if the checkbox above is checked.")
        else:
            for backup in backups:
                bcol1, bcol2 = st.columns([3, 1])
                bcol1.caption(f"{backup['filename']} — saved {backup['modified']}")
                if bcol2.button("Restore", key=f"restore_{to_train.name}_{backup['filename']}"):
                    onboard.restore_model_backup(to_train.name, backup["path"], registry=registry)
                    st.success(f"Restored {backup['filename']} as the active model.")
                    st.rerun()

    st.markdown(f"**Training history ({to_train.display_name})**")
    training_history = training_runs_store.list_for_component(to_train.name)
    if not training_history:
        st.caption("No training runs recorded yet.")
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "when": run.created_at,
                        "source": run.source,
                        "status": run.status,
                        "threshold": run.threshold,
                        "error": run.error,
                        **run.metrics,
                    }
                    for run in training_history
                ]
            ),
            width="stretch",
        )
        with st.expander("Full performance data for a specific run"):
            run_by_label = {
                f"#{run.id} — {run.created_at} ({run.source}, {run.status})": run for run in training_history
            }
            picked_label = st.selectbox(
                "Run", list(run_by_label.keys()), key=f"training_history_pick_{to_train.name}"
            )
            picked_run = run_by_label[picked_label]
            if picked_run.error:
                st.error(picked_run.error)
            st.json({"settings": picked_run.settings, "metrics": picked_run.metrics, "details": picked_run.details})

            if picked_run.evaluation_dir:
                st.markdown(f"**Evaluation report** — `{picked_run.evaluation_dir}`")
                eval_path = for_component(to_train.name).root / picked_run.evaluation_dir
                plot_paths = sorted(eval_path.glob("*.png")) if eval_path.exists() else []
                if plot_paths:
                    # Two columns so a report with several plots (classifier:
                    # confusion matrix + loss + accuracy; YOLO: mAP + loss +
                    # confusion matrix + PR/F1/P/R) doesn't turn into one very
                    # long single-column scroll.
                    cols = st.columns(2)
                    for i, plot_path in enumerate(plot_paths):
                        cols[i % 2].image(str(plot_path), caption=plot_path.name, width="stretch")
                else:
                    st.caption("No plot files in this run's evaluation report (see notes above).")
            elif picked_run.status == "success":
                st.caption(
                    "No evaluation report for this run — generate_evaluation_report was off at the "
                    "time, or nothing meaningful could be evaluated for this model_type."
                )

    with st.expander("Adjust settings before training"):
        st.caption(
            "Original training files are kept at full resolution on disk and re-read at "
            "training time, so raising image_size here works immediately without "
            "re-uploading anything."
        )
        ecol1, ecol2 = st.columns(2)
        edit_image_size = ecol1.number_input(
            "Image size",
            min_value=32,
            step=16,
            value=to_train.image_size,
            key=f"edit_image_size_{to_train.name}",
        )
        edit_epochs = ecol2.number_input(
            "Epochs", min_value=1, value=to_train.epochs, key=f"edit_epochs_{to_train.name}"
        )
        edit_latent_dim = ecol1.number_input(
            "Latent dimension",
            min_value=2,
            value=to_train.latent_dim,
            key=f"edit_latent_dim_{to_train.name}",
        )
        edit_batch_size = ecol2.number_input(
            "Batch size", min_value=1, value=to_train.batch_size, key=f"edit_batch_size_{to_train.name}"
        )

        with st.expander("Advanced settings"):
            current_settings = {key: getattr(to_train, key) for key in SETTINGS_DEFAULTS}
            edit_advanced = _render_advanced_settings(current_settings, key_prefix=f"edit_{to_train.name}")

        with st.expander("Reporting (optional)"):
            edit_reporting = _render_reporting_settings(current_settings, key_prefix=f"edit_{to_train.name}")

    if st.button("Train", type="primary"):
        registry.update_settings(
            to_train.name,
            image_size=int(edit_image_size),
            epochs=int(edit_epochs),
            latent_dim=int(edit_latent_dim),
            batch_size=int(edit_batch_size),
            **edit_advanced,
            **edit_reporting,
        )
        if backup_before_training:
            backup_path = onboard.backup_current_model(to_train.name, registry=registry)
            if backup_path:
                st.caption(f"Backed up current model to `{backup_path}` before training.")
        if selected_correction_ids:
            incorporation = onboard.incorporate_verified_corrections(
                to_train.name, selected_correction_ids, registry=registry
            )
            if incorporation.incorporated:
                st.success(
                    f"Incorporated {incorporation.incorporated} verified correction(s) "
                    f"({incorporation.images_added} image copy/copies) into training/ — permanent, "
                    "picked up by this and every future training run."
                )
                if incorporation.per_bucket_counts:
                    st.caption(
                        "Per-bucket counts: "
                        + ", ".join(f"{bucket}: {count}" for bucket, count in sorted(incorporation.per_bucket_counts.items()))
                    )
            else:
                st.caption("No corrections were actually incorporated this run.")
            if incorporation.skipped_unusable:
                st.caption(
                    f"{incorporation.skipped_unusable} correction(s) skipped as not usable this way "
                    "(see logs for details) — still pending, offered again next time."
                )
        with st.spinner(f"Training '{to_train.display_name}'... this may take a while."):
            try:
                result = onboard.train_component(
                    to_train.name,
                    registry=registry,
                    # Traceability only (see train_component()'s own docstring) —
                    # links this run's evaluation report back to "which
                    # corrections was this retrain responding to", so a
                    # before/after comparison (Training history below) can show
                    # whether they actually helped. Only passed if corrections
                    # were actually incorporated this round, not just selected.
                    incorporated_correction_ids=(
                        selected_correction_ids if selected_correction_ids and incorporation.incorporated else None
                    ),
                )
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                st.caption("This failed attempt was still recorded in the training history above.")
            else:
                if result.threshold is not None:
                    st.success(f"Training complete. Threshold set to {result.threshold:.6f}")
                else:
                    st.success("Training complete.")
                st.caption("Saved to this component's training history above.")

                summary = {
                    k: v
                    for k, v in result.details.items()
                    if k not in ("approved_errors", "failed_errors", "history")
                }
                if summary:
                    st.json(summary)

                approved_errors = result.details.get("approved_errors")
                failed_errors = result.details.get("failed_errors")
                rows = [{"error": e, "set": "approved"} for e in (approved_errors or [])]
                rows += [{"error": e, "set": "failed"} for e in (failed_errors or [])]

                if rows and result.threshold is not None:
                    df = pd.DataFrame(rows)
                    points = alt.Chart(df).mark_tick(thickness=2, size=30).encode(
                        x=alt.X("error:Q", title="Reconstruction error"),
                        y=alt.Y("set:N", title=None),
                        color="set:N",
                    )
                    rule = (
                        alt.Chart(pd.DataFrame({"threshold": [result.threshold]}))
                        .mark_rule(color="red", strokeDash=[4, 4])
                        .encode(x="threshold:Q")
                    )
                    st.altair_chart(points + rule, width="stretch")

st.divider()
st.subheader("Reporting: knowledge base & machine parameters")
st.caption(
    "Only relevant for components with 'Enable AI-generated reports' turned on above. Knowledge "
    "documents and machine-parameter definitions are both optional and independent of each other — "
    "a component can have neither, either, or both; reporting degrades honestly when something "
    "isn't configured (see core/reporting/reporter.py) rather than requiring all-or-nothing."
)

reporting_components = [c for c in registry.list_all() if c.reporting_enabled]
if not reporting_components:
    st.info("No components have reporting enabled yet — enable it above (in a component's 'Reporting' section) first.")
else:
    report_target = st.selectbox(
        "Component",
        reporting_components,
        format_func=lambda c: f"{c.display_name} ({c.status})",
        key="report_mgmt_target",
    )
    report_paths = for_component(report_target.name)

    kcol1, kcol2 = st.columns(2)

    with kcol1:
        st.markdown("**Knowledge documents**")
        existing_docs = (
            sorted(p.name for p in report_paths.knowledge_dir.iterdir() if p.is_file())
            if report_paths.knowledge_dir.exists()
            else []
        )
        st.caption(f"Currently on disk: {', '.join(existing_docs) if existing_docs else '(none)'}")
        uploaded_docs = st.file_uploader(
            "Upload .md/.txt documents",
            type=["md", "txt"],
            accept_multiple_files=True,
            key=f"report_docs_{report_target.name}",
            help=(
                "Each file needs its own YAML frontmatter (doc_type: manual/spec/incident/"
                "machine_context, and source) — see data/components/tandborste/knowledge/ for "
                "examples of the expected format. Uploading doesn't overwrite existing files unless "
                "the name matches exactly."
            ),
        )
        if st.button(
            "Save & index documents", key=f"report_docs_save_{report_target.name}", disabled=not uploaded_docs
        ):
            report_paths.knowledge_dir.mkdir(parents=True, exist_ok=True)
            for f in uploaded_docs:
                (report_paths.knowledge_dir / f.name).write_bytes(f.getvalue())
            try:
                count = reporting_indexer.index_component_type(report_target.name)
            except Exception as exc:
                st.error(f"Saved the file(s), but indexing failed: {exc}")
            else:
                st.success(f"Saved {len(uploaded_docs)} file(s), indexed {count} chunk(s).")
                st.rerun()

    with kcol2:
        st.markdown("**Machine parameters**")
        state_key = f"report_params_{report_target.name}"
        id_counter_key = f"{state_key}_next_id"
        if state_key not in st.session_state:
            st.session_state[state_key] = [
                {"_id": i, **{k: v for k, v in vars(p).items()}}
                for i, p in enumerate(parse_machine_parameters(report_target.machine_parameters))
            ]
            st.session_state[id_counter_key] = len(st.session_state[state_key])

        # Each row's widget keys are tied to a stable "_id" assigned once
        # when the row is added, never to its list position — a plain
        # text_input does not reliably pick up a fresh `value=` after a
        # session_state pop or a position shift (confirmed the hard way
        # while building the grid search UI's metric-key field), so a row
        # deleted/reordered above another must not cause the widget below
        # it to silently keep showing stale text tied to the old position.
        rows = st.session_state[state_key]
        to_delete = None
        for row in rows:
            rid = row["_id"]
            pcol1, pcol2, pcol3, pcol4, pcol5, pcol6, pcol7 = st.columns([2, 1, 1, 1, 2, 2, 0.6])
            row["name"] = pcol1.text_input("Name", value=row["name"], key=f"{state_key}_{rid}_name")
            row["unit"] = pcol2.text_input("Unit", value=row["unit"], key=f"{state_key}_{rid}_unit")
            row["normal_min"] = pcol3.number_input(
                "Min", value=float(row["normal_min"]), key=f"{state_key}_{rid}_min"
            )
            row["normal_max"] = pcol4.number_input(
                "Max", value=float(row["normal_max"]), key=f"{state_key}_{rid}_max"
            )
            row["above_state"] = pcol5.text_input(
                "Above-normal state", value=row["above_state"], key=f"{state_key}_{rid}_above"
            )
            row["below_state"] = pcol6.text_input(
                "Below-normal state (optional)", value=row["below_state"] or "", key=f"{state_key}_{rid}_below"
            )
            pcol7.write("")  # vertical alignment for the delete button against labeled inputs above
            if pcol7.button("Remove", key=f"{state_key}_{rid}_del"):
                to_delete = rid
        if to_delete is not None:
            st.session_state[state_key] = [r for r in rows if r["_id"] != to_delete]
            st.rerun()

        if st.button("+ Add parameter", key=f"{state_key}_add"):
            st.session_state[state_key].append(
                {
                    "_id": st.session_state[id_counter_key],
                    "name": "",
                    "unit": "",
                    "normal_min": 0.0,
                    "normal_max": 0.0,
                    "above_state": "",
                    "below_state": "",
                }
            )
            st.session_state[id_counter_key] += 1
            st.rerun()

        if st.button("Save parameter definitions", key=f"{state_key}_save", type="primary"):
            defs = [
                MachineParameterDef(
                    name=row["name"],
                    unit=row["unit"],
                    normal_min=float(row["normal_min"]),
                    normal_max=float(row["normal_max"]),
                    above_state=row["above_state"],
                    below_state=row["below_state"] or None,
                )
                for row in st.session_state[state_key]
                if row["name"].strip()
            ]
            registry.update_settings(report_target.name, machine_parameters=serialize_machine_parameters(defs))
            st.success(f"Saved {len(defs)} parameter definition(s).")

st.divider()
st.subheader("Grid search")
st.caption(
    "Sweep a grid of settings, training once per combination against an isolated scratch copy "
    "of the component's data — the component you search is never touched (its settings, "
    "status, and trained model stay exactly as they are). Results are ranked by the metric key "
    "below, read from each trial's TrainResult.details['metrics'] (pre-filled with a sensible "
    "default per analysis method, but editable — exact key names come from the underlying "
    "library and can vary). Apply a winning configuration yourself via 'Adjust settings before "
    "training' above, then retrain."
)

components_for_search = registry.list_all()
if not components_for_search:
    st.info("No components yet. Create one above first.")
else:
    _ALL_GRID_SEARCH_WIDGET_KEYS = [f"gs_{key}" for key in grid_search.PARAM_SPECS]

    def _reset_grid_search_widgets() -> None:
        # A keyed widget's `default=` is only honored on its very first
        # render — on every rerun after that, its own session_state (tied to
        # its key) wins. Without this, switching "Component to search" would
        # leave every widget showing (and sweeping from) whichever component
        # was selected *before*, not the newly picked one — same class of
        # bug already found and fixed for the annotate/train selectboxes
        # elsewhere on this page. Every possible gs_* key is cleared
        # regardless of the newly selected component's model_type, since
        # only a subset renders for any given one anyway. The metric-key
        # text inputs aren't here — they're keyed per-component instead
        # (see below), since popping alone doesn't reliably refresh a
        # text_input's displayed value.
        for widget_key in _ALL_GRID_SEARCH_WIDGET_KEYS:
            st.session_state.pop(widget_key, None)

    search_target = st.selectbox(
        "Component to search",
        components_for_search,
        format_func=lambda c: f"{c.display_name} ({c.status}, {c.model_type})",
        key="grid_search_target",
        on_change=_reset_grid_search_widgets,
    )

    def _grid_options(preset: list, current):
        """Preset candidate values, plus the component's current value if it isn't already in there."""
        return sorted({*preset, current}, key=str)

    sweepable_keys = grid_search.SWEEPABLE_SETTINGS_BY_MODEL_TYPE.get(search_target.model_type, [])
    st.caption("Pick one or more values per setting — a setting left at a single value stays fixed.")

    param_grid: dict[str, list] = {}
    for i in range(0, len(sweepable_keys), 2):
        row_keys = sweepable_keys[i : i + 2]
        row_cols = st.columns(2)
        for setting_key, col in zip(row_keys, row_cols):
            spec = grid_search.PARAM_SPECS[setting_key]
            current = getattr(search_target, setting_key)
            widget_key = f"gs_{setting_key}"
            if spec["kind"] == "select":
                chosen = col.multiselect(
                    spec["label"],
                    _grid_options(spec["options"], current),
                    default=[current],
                    key=widget_key,
                    help=spec.get("help"),
                )
            elif spec["kind"] == "checkbox":
                chosen = col.multiselect(
                    spec["label"],
                    [True, False],
                    default=[bool(current)],
                    key=widget_key,
                    help=spec.get("help"),
                )
            else:  # "number"
                chosen = col.multiselect(
                    spec["label"],
                    _grid_options(spec["presets"], current),
                    default=[current],
                    key=widget_key,
                    help=spec.get("help"),
                )
            param_grid[setting_key] = chosen or [current]

    default_metric_key, default_fallback_key = grid_search.DEFAULT_METRIC_KEY_BY_MODEL_TYPE.get(
        search_target.model_type, ("", None)
    )
    mcol1, mcol2 = st.columns(2)
    # Keyed by component name, not a fixed string: unlike st.selectbox/
    # st.multiselect (which pick up a popped session_state key's fresh
    # `default=`/`index=` immediately), st.text_input does not reliably
    # re-render with a new `value=` after just a session_state pop — its
    # frontend keeps showing the last-typed/last-rendered text unless the
    # widget gets a genuinely new key (confirmed by direct testing: pop()
    # alone left this showing the previous component's metric key). Tying
    # the key to the component forces a fresh widget identity whenever
    # "Component to search" changes, which does reliably pick up the new
    # default.
    metric_key = mcol1.text_input(
        "Metric key",
        value=default_metric_key,
        key=f"gs_metric_key_{search_target.name}",
        help="Looked up in each trial's TrainResult.details['metrics']. Trials are ranked by this value, highest first.",
    )
    fallback_metric_key = mcol2.text_input(
        "Fallback metric key (optional)",
        value=default_fallback_key or "",
        key=f"gs_fallback_metric_key_{search_target.name}",
        help="Used only if the metric key above is missing from a trial's metrics.",
    )

    total_combinations = grid_search.count_combinations(param_grid)
    st.caption(f"This will run **{total_combinations}** training run(s), one per combination.")

    MANY_COMBINATIONS_THRESHOLD = 30
    run_allowed = True
    if total_combinations > MANY_COMBINATIONS_THRESHOLD:
        st.warning(
            f"{total_combinations} combinations could take a very long time (each is a full "
            "training run). Consider narrowing your selection, or confirm below to run it anyway."
        )
        run_allowed = st.checkbox("I understand this may take a long time — run it anyway")
    if not metric_key.strip():
        run_allowed = False
        st.warning("Metric key is required.")

    if st.button("Run grid search", type="primary", disabled=not run_allowed):
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        results_table = st.empty()
        rows: list[dict] = []

        def _on_trial_complete(done: int, total: int, trial: grid_search.Trial) -> None:
            progress_bar.progress(done / total)
            status_text.text(f"Trial {done}/{total}: {trial.settings}")
            rows.append({**trial.settings, "metric": trial.metric_value, "error": trial.error})
            results_table.dataframe(
                pd.DataFrame(rows).sort_values(
                    "metric", ascending=False, na_position="last", kind="stable"
                ),
                width="stretch",
            )

        with st.spinner(f"Running {total_combinations} training run(s)..."):
            try:
                trials = grid_search.run_grid_search(
                    search_target.name,
                    param_grid,
                    metric_key=metric_key.strip(),
                    fallback_metric_key=fallback_metric_key.strip() or None,
                    registry=registry,
                    on_trial_complete=_on_trial_complete,
                )
            except Exception as exc:
                st.error(f"Grid search failed: {exc}")
                trials = []

        if trials:
            status_text.empty()
            progress_bar.empty()
            best = trials[0]
            if best.metric_value is not None:
                st.success(
                    f"Grid search complete — {len(trials)} run(s). Best: {best.settings} "
                    f"({metric_key.strip()}={best.metric_value:.4f}). Apply these values yourself "
                    "via 'Adjust settings before training' above, then retrain."
                )
            else:
                st.warning(
                    f"Grid search complete — {len(trials)} run(s), but none produced a usable "
                    f"'{metric_key.strip()}' value (check the 'error' column in the results above)."
                )
