"""Generic grid search over a component's training settings.

Works through the same BaseTrainer interface every method already
implements — no method-specific training logic lives here. Each trial runs
against a *scratch* copy of the component (same modality/model_type,
training data copied once at the start of the sweep) so the source
component's own settings, status, and trained model artifact are never
touched — you review the results and apply a winning configuration to the
real component yourself (via the existing "Adjust settings before
training" flow), rather than this silently overwriting anything.

Built especially for YOLO — its metrics (mAP50, precision, recall, fitness)
are already flat floats in `TrainResult.details["metrics"]`, see
`core/detection/yolo/trainer.py` — but nothing here is YOLO-specific: any
model_type whose `TrainResult.details["metrics"]` contains the requested
`metric_key` works the same way.
"""

from __future__ import annotations

import itertools
import shutil
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from emil_ml.config.registry import SETTINGS_DEFAULTS, ComponentRegistry
from emil_ml.config.settings import (
    VALID_CLASS_WEIGHT_STRATEGIES,
    VALID_CLASSIFIER_BASE_MODELS,
    VALID_CLASSIFIER_POOLING_TYPES,
    VALID_ISOLATION_FOREST_CONTAMINATION_OPTIONS,
    VALID_PATCHCORE_BACKBONES,
    VALID_SCORE_METHODS,
    VALID_YOLO_MODEL_VARIANTS,
    VALID_YOLO_OPTIMIZERS,
)
from emil_ml.core import registry_factory
from emil_ml.core.training_runs import store as training_runs_store
from emil_ml.utils.paths import for_component, slugify

_SCRATCH_LABEL_SUFFIX = " (grid search scratch)"

# Which SETTINGS_DEFAULTS keys are actually read by each model_type's
# trainer — e.g. PatchCore never touches image_size/epochs, Isolation Forest
# uses none of the "shared" training fields at all (it fits a forest on
# embeddings in one shot, no epoch loop). decision_rule is deliberately
# excluded from YOLO's list: it only changes how a prediction is interpreted
# at inference time, never training itself, so sweeping it can't affect any
# metric a grid search would rank by. Drives which sweep widgets the Onboard
# page's Grid Search section renders for a given component.
SWEEPABLE_SETTINGS_BY_MODEL_TYPE: dict[str, list[str]] = {
    "autoencoder": [
        "image_size",
        "epochs",
        "latent_dim",
        "batch_size",
        "early_stopping_patience",
        "score_method",
        "threshold_percentile",
    ],
    "classifier": [
        "image_size",
        "epochs",
        "batch_size",
        "early_stopping_patience",
        "base_model",
        "pooling",
        "class_weight_strategy",
        "augmentation_strength",
        "fine_tune_epochs",
        "fine_tune_learning_rate",
        "fine_tune_unfreeze_layers",
    ],
    "yolo": [
        "yolo_model_variant",
        "yolo_mosaic",
        "yolo_class_loss_weight",
        "yolo_augmentation_strength",
        "image_size",
        "batch_size",
        "epochs",
        "early_stopping_patience",
        "yolo_optimizer",
        "yolo_learning_rate",
    ],
    "patchcore": [
        "patchcore_backbone",
        "patchcore_coreset_sampling_ratio",
        "patchcore_num_neighbors",
        "batch_size",
    ],
    "isolation_forest": [
        "isolation_forest_n_estimators",
        "isolation_forest_contamination",
        "isolation_forest_max_features",
        "isolation_forest_standardize",
    ],
}

# UI hints for how to render a sweep widget for each setting: "select" (a
# fixed set of valid string values), "number" (numeric presets — the
# component's *current* value is always added too, see `_grid_options` in
# the Onboard page), or "checkbox" (boolean). Every key referenced by
# SWEEPABLE_SETTINGS_BY_MODEL_TYPE above must have an entry here.
PARAM_SPECS: dict[str, dict[str, Any]] = {
    "image_size": {"kind": "number", "label": "Image size", "presets": [128, 256, 512, 640, 1024]},
    "epochs": {"kind": "number", "label": "Epochs", "presets": [10, 25, 50, 100, 150, 200, 300, 500]},
    "batch_size": {"kind": "number", "label": "Batch size", "presets": [4, 8, 16, 32, 64]},
    "latent_dim": {"kind": "number", "label": "Latent dimension", "presets": [32, 64, 128, 256, 512]},
    "early_stopping_patience": {
        "kind": "number",
        "label": "Early stopping patience",
        "presets": [0, 4, 8, 16, 32, 50, 100, 200],
    },
    "score_method": {"kind": "select", "label": "Score method", "options": list(VALID_SCORE_METHODS)},
    "threshold_percentile": {
        "kind": "number",
        "label": "Threshold percentile",
        "presets": [90.0, 95.0, 97.5, 99.0, 99.5],
    },
    "base_model": {"kind": "select", "label": "Base model", "options": list(VALID_CLASSIFIER_BASE_MODELS)},
    "pooling": {"kind": "select", "label": "Pooling", "options": list(VALID_CLASSIFIER_POOLING_TYPES)},
    "class_weight_strategy": {
        "kind": "select",
        "label": "Class weight strategy",
        "options": list(VALID_CLASS_WEIGHT_STRATEGIES),
    },
    "augmentation_strength": {
        "kind": "number",
        "label": "Augmentation strength",
        "presets": [0.0, 0.1, 0.2, 0.3, 0.5],
    },
    "fine_tune_epochs": {"kind": "number", "label": "Fine-tune epochs", "presets": [5, 10, 20, 30]},
    "fine_tune_learning_rate": {
        "kind": "number",
        "label": "Fine-tune LR",
        "presets": [0.00001, 0.00005, 0.0001, 0.0005],
    },
    "fine_tune_unfreeze_layers": {
        "kind": "number",
        "label": "Unfreeze layers",
        "presets": [0, 10, 20, 40],
    },
    "yolo_model_variant": {
        "kind": "select",
        "label": "YOLO model variant",
        "options": list(VALID_YOLO_MODEL_VARIANTS),
    },
    "yolo_mosaic": {"kind": "number", "label": "Mosaic augmentation", "presets": [0.0, 0.25, 0.5, 0.75, 1.0]},
    "yolo_class_loss_weight": {
        "kind": "number",
        "label": "Class loss weight",
        "presets": [0.25, 0.3, 0.35, 0.4, 0.5, 1.0, 2.0],
    },
    "yolo_augmentation_strength": {
        "kind": "number",
        "label": "Geometric/color augmentation strength",
        "presets": [0.0, 0.25, 0.5, 0.75, 1.0],
    },
    "yolo_optimizer": {
        "kind": "select",
        "label": "Optimizer",
        "options": list(VALID_YOLO_OPTIMIZERS),
        "help": "'auto' ignores the learning rate values below entirely — see the Optimizer setting's own help text above for why.",
    },
    "yolo_learning_rate": {
        "kind": "number",
        "label": "Learning rate (lr0)",
        "presets": [0.001, 0.005, 0.01, 0.02],
    },
    "patchcore_backbone": {"kind": "select", "label": "Backbone", "options": list(VALID_PATCHCORE_BACKBONES)},
    "patchcore_coreset_sampling_ratio": {
        "kind": "number",
        "label": "Coreset sampling ratio",
        "presets": [0.05, 0.1, 0.25, 0.5, 1.0],
    },
    "patchcore_num_neighbors": {"kind": "number", "label": "Num neighbors", "presets": [1, 3, 5, 9, 15]},
    "isolation_forest_n_estimators": {
        "kind": "number",
        "label": "Number of trees",
        "presets": [50, 100, 200, 300],
    },
    "isolation_forest_contamination": {
        "kind": "select",
        "label": "Contamination",
        "options": list(VALID_ISOLATION_FOREST_CONTAMINATION_OPTIONS),
    },
    "isolation_forest_max_features": {
        "kind": "number",
        "label": "Max features",
        "presets": [0.25, 0.5, 0.75, 1.0],
    },
    "isolation_forest_standardize": {"kind": "checkbox", "label": "Standardize embeddings"},
}

# (metric_key, fallback_metric_key) to pre-fill in the Onboard page's Grid
# Search UI, per model_type — see each trainer's TrainResult.details["metrics"]
# for where these come from. Always editable in the UI, not enforced here:
# exact key names are supplied by the underlying library (Ultralytics,
# anomalib) and can drift between versions.
DEFAULT_METRIC_KEY_BY_MODEL_TYPE: dict[str, tuple[str, str | None]] = {
    "yolo": ("metrics/mAP50(B)", "fitness"),
    "patchcore": ("image_AUROC", "image_F1Score"),
    "isolation_forest": ("roc_auc", "failed_flagged_ratio"),
    "autoencoder": ("failed_flagged_ratio", None),
    "classifier": ("val_balanced_accuracy", "val_accuracy"),
}


@dataclass(frozen=True)
class Trial:
    """One grid search combination's outcome."""

    settings: dict[str, Any]
    metric_value: float | None  # None if the metric key was missing or the run raised
    error: str | None = None  # set only if training itself raised
    details: dict[str, Any] = field(default_factory=dict)


def count_combinations(param_grid: dict[str, list[Any]]) -> int:
    """How many trials `run_grid_search` would run for this grid — for UI display before committing."""
    total = 1
    for values in param_grid.values():
        total *= max(len(values), 1)
    return total


def _combinations(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not param_grid:
        return [{}]
    keys = list(param_grid.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*(param_grid[k] for k in keys))]


def _copy_training_data(source_name: str, scratch_name: str, model_type: str) -> None:
    """Copy the source component's training data into the scratch component's directories.

    YOLO's data lives in its annotation pool (images + labels + classes.txt);
    every other method uses the approved/failed folder convention. Both
    directory layouts are derived purely from a component's name (see
    utils/paths.py), so this is the only place that needs to know which one
    applies — the trainer itself doesn't care how its data got there.
    """
    source_paths = for_component(source_name)
    scratch_paths = for_component(scratch_name)
    scratch_paths.create_all()

    if model_type == "yolo":
        if source_paths.yolo_images_dir.exists():
            shutil.copytree(source_paths.yolo_images_dir, scratch_paths.yolo_images_dir, dirs_exist_ok=True)
        if source_paths.yolo_labels_dir.exists():
            shutil.copytree(source_paths.yolo_labels_dir, scratch_paths.yolo_labels_dir, dirs_exist_ok=True)
        if source_paths.yolo_classes_file.exists():
            shutil.copy2(source_paths.yolo_classes_file, scratch_paths.yolo_classes_file)
    else:
        if source_paths.training_approved_dir.exists():
            shutil.copytree(
                source_paths.training_approved_dir, scratch_paths.training_approved_dir, dirs_exist_ok=True
            )
        if source_paths.training_failed_dir.exists():
            shutil.copytree(
                source_paths.training_failed_dir, scratch_paths.training_failed_dir, dirs_exist_ok=True
            )


def run_grid_search(
    source_component_name: str,
    param_grid: dict[str, list[Any]],
    *,
    metric_key: str,
    fallback_metric_key: str | None = None,
    registry: ComponentRegistry | None = None,
    on_trial_complete: Callable[[int, int, Trial], None] | None = None,
) -> list[Trial]:
    """Train one combination per grid cell against a scratch copy of `source_component_name`'s data.

    Ranks results by `metric_value` descending (None sorts last, so a failed
    trial never accidentally looks like a winner). `on_trial_complete(done,
    total, trial)`, if given, fires after each trial — the Streamlit UI uses
    this to update a live progress bar/results table across a single long
    call, rather than only being able to show something after the whole
    sweep finishes.

    The scratch component is deleted (registry row + all files) once the
    sweep ends, including if it's interrupted by an exception, so a crashed
    sweep doesn't leave orphaned scratch data behind.
    """
    registry = registry or ComponentRegistry()
    source = registry.get(source_component_name)
    if source is None:
        raise KeyError(f"No component named {source_component_name!r}")

    combinations = _combinations(param_grid)

    scratch_display_name = f"{source.display_name}{_SCRATCH_LABEL_SUFFIX}"
    scratch_name = slugify(scratch_display_name)
    if registry.get(scratch_name):
        registry.delete(scratch_name)
        shutil.rmtree(for_component(scratch_name).root, ignore_errors=True)

    created = registry.create(scratch_display_name, modality=source.modality, model_type=source.model_type)
    assert created.name == scratch_name  # slugify() is deterministic — this should always hold
    # Clone every one of the source's *current* settings onto the scratch
    # component — not just modality/model_type. Without this, any setting
    # a given param_grid doesn't happen to sweep (or a caller sweeps only
    # some of) would silently train with the global default instead of the
    # source's actual configured value (confirmed: epochs reverted to the
    # default 50 instead of a source component's actual 1, in a sweep that
    # only varied image_size and optimizer).
    registry.update_settings(scratch_name, **{key: getattr(source, key) for key in SETTINGS_DEFAULTS})
    _copy_training_data(source.name, scratch_name, source.model_type)

    # Groups every trial from this one sweep together in training_runs — see
    # that table's own docstring. Logged against `source.name`, not
    # `scratch_name`: the scratch component and its files are deleted at the
    # end of this function (below), so a row that outlives the sweep needs
    # to be attributed to the real component a human was actually exploring
    # settings for, not the throwaway copy that happened to produce it.
    batch_id = str(uuid.uuid4())

    trials: list[Trial] = []
    try:
        trainer = registry_factory.get_trainer(source.modality, source.model_type)
        for i, combo in enumerate(combinations):
            registry.update_settings(scratch_name, **combo)
            scratch_component = registry.get(scratch_name)
            trial_settings = {key: getattr(scratch_component, key) for key in SETTINGS_DEFAULTS}
            try:
                result = trainer.train(scratch_component)
                metrics = result.details.get("metrics", {})
                value = metrics.get(metric_key)
                if value is None and fallback_metric_key:
                    value = metrics.get(fallback_metric_key)
                trial = Trial(settings=combo, metric_value=value, details=result.details)
                training_runs_store.create(
                    source.name,
                    display_name=source.display_name,
                    modality=source.modality,
                    model_type=source.model_type,
                    source="grid_search",
                    status="success",
                    threshold=result.threshold,
                    settings=trial_settings,
                    metrics=metrics,
                    details=result.details,
                    grid_search_batch_id=batch_id,
                )
            except Exception as exc:  # noqa: BLE001 - one bad combination shouldn't kill the whole sweep
                trial = Trial(settings=combo, metric_value=None, error=str(exc))
                training_runs_store.create(
                    source.name,
                    display_name=source.display_name,
                    modality=source.modality,
                    model_type=source.model_type,
                    source="grid_search",
                    status="failed",
                    error=str(exc),
                    settings=trial_settings,
                    grid_search_batch_id=batch_id,
                )
            trials.append(trial)
            if on_trial_complete:
                on_trial_complete(i + 1, len(combinations), trial)
    finally:
        registry.delete(scratch_name)
        shutil.rmtree(for_component(scratch_name).root, ignore_errors=True)

    trials.sort(key=lambda t: (t.metric_value is None, -(t.metric_value or 0.0)))
    return trials
