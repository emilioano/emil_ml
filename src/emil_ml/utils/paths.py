"""Deterministic per-component filesystem paths.

Folder layout for a component is derived purely from its slug `name`, so the
database never needs to store paths.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from emil_ml.config.settings import (
    ANALYZED_SUBDIR,
    APPROVED_SUBDIR,
    ARCHIVE_SUBDIR,
    AUTOENCODER_MODEL_FILENAME,
    CLASSIFIER_MODEL_FILENAME,
    COMPONENTS_DIR,
    ERROR_SUBDIR,
    EVALUATION_SUBDIR,
    FAILED_SUBDIR,
    INPUT_SUBDIR,
    ISOLATION_FOREST_MODEL_FILENAME,
    ISOLATION_FOREST_SCALER_FILENAME,
    KNOWLEDGE_SUBDIR,
    MODELS_SUBDIR,
    PATCHCORE_MODEL_FILENAME,
    TRAINING_SUBDIR,
    YOLO_ANNOTATIONS_SUBDIR,
    YOLO_DATASET_SUBDIR,
    YOLO_MODEL_FILENAME,
)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(display_name: str) -> str:
    """Convert a display name into a URL/filesystem-safe slug."""
    slug = _SLUG_RE.sub("-", display_name.strip().lower()).strip("-")
    if not slug:
        raise ValueError(f"Cannot derive a valid slug from {display_name!r}")
    return slug


@dataclass(frozen=True)
class ComponentPaths:
    """All filesystem paths for a single component, derived from its name."""

    name: str

    @property
    def root(self) -> Path:
        return COMPONENTS_DIR / self.name

    @property
    def training_dir(self) -> Path:
        return self.root / TRAINING_SUBDIR

    @property
    def training_approved_dir(self) -> Path:
        return self.training_dir / APPROVED_SUBDIR

    @property
    def training_failed_dir(self) -> Path:
        return self.training_dir / FAILED_SUBDIR

    @property
    def models_dir(self) -> Path:
        return self.root / MODELS_SUBDIR

    def model_file(self, filename: str) -> Path:
        """Path to a model artifact within this component's models/ directory."""
        return self.models_dir / filename

    @property
    def evaluation_dir(self) -> Path:
        """Root of every training run's evaluation artifacts — see
        evaluation_run_dir() for where one specific run's files actually
        land. Sits beside models_dir, not inside it: evaluation artifacts
        (plots, metrics.json) aren't a model file a predictor loads, they're
        a human-facing report about one training run."""
        return self.root / EVALUATION_SUBDIR

    def evaluation_run_dir(self, timestamp: str) -> Path:
        """One training run's evaluation artifacts — timestamped, never
        overwritten (see training/onboard.py's train_component(), the only
        writer), so a before/after comparison across retrains (e.g. after
        incorporating verified corrections) is always possible: every past
        run's artifacts stay exactly as they were. Not part of all_dirs()/
        create_all() — like archive_dir_for_date(), this only comes into
        existence the first time a training run actually generates a
        report, and there's an unbounded number of them to pre-create
        anyway."""
        return self.evaluation_dir / timestamp

    def resolve_model_path(self, stored_model_path: str) -> Path:
        """Resolve a component's primary model file from its DB-stored,
        root-relative `Component.model_path` (see training/onboard.py,
        which is what writes it — always relative going forward, e.g.
        "models/autoencoder.keras"). This is what makes model_path an
        actual functioning pointer rather than a redundant cache of
        something re-derivable from name + model_type: swap it to a
        different filename under models/ (e.g. to keep multiple trained
        versions and alternate between them) and predictors follow it,
        no code change needed.

        Accepts an already-absolute path unchanged, for backward
        compatibility with rows written before this field was
        root-relative.
        """
        candidate = Path(stored_model_path)
        return candidate if candidate.is_absolute() else self.root / candidate

    @property
    def model_path(self) -> Path:
        """Autoencoder's model path. Kept for backward compatibility; prefer model_file()."""
        return self.model_file(AUTOENCODER_MODEL_FILENAME)

    @property
    def classifier_model_path(self) -> Path:
        return self.model_file(CLASSIFIER_MODEL_FILENAME)

    @property
    def yolo_model_path(self) -> Path:
        return self.model_file(YOLO_MODEL_FILENAME)

    @property
    def yolo_model_backup_path(self) -> Path:
        """The model this component's most recent retrain replaced, kept as a
        single rolling backup (see YoloTrainer.train()) — not versioned
        history, just a one-run-back safety net."""
        return self.model_file(f"{Path(YOLO_MODEL_FILENAME).stem}.bak{Path(YOLO_MODEL_FILENAME).suffix}")

    @property
    def patchcore_model_path(self) -> Path:
        return self.model_file(PATCHCORE_MODEL_FILENAME)

    @property
    def patchcore_results_dir(self) -> Path:
        """Scratch dir anomalib's Engine writes Lightning logs/checkpoints into —
        the final checkpoint is copied out to patchcore_model_path; this is
        rebuilt fresh every training run, same spirit as yolo_dataset_dir."""
        return self.root / "patchcore_results"

    @property
    def isolation_forest_model_path(self) -> Path:
        return self.model_file(ISOLATION_FOREST_MODEL_FILENAME)

    @property
    def isolation_forest_scaler_path(self) -> Path:
        """Only exists on disk when the model was trained with standardize on."""
        return self.model_file(ISOLATION_FOREST_SCALER_FILENAME)

    # --- YOLO annotation pool: persistent source of truth regardless of which
    # of the three annotation paths (pre-made labels, mask conversion, manual
    # drawing) produced it. One image + one label .txt per image (empty file
    # = no boxes = negative example), plus a shared classes.txt.
    @property
    def yolo_annotations_dir(self) -> Path:
        return self.training_dir / YOLO_ANNOTATIONS_SUBDIR

    @property
    def yolo_images_dir(self) -> Path:
        return self.yolo_annotations_dir / "images"

    @property
    def yolo_labels_dir(self) -> Path:
        return self.yolo_annotations_dir / "labels"

    @property
    def yolo_classes_file(self) -> Path:
        return self.yolo_annotations_dir / "classes.txt"

    @property
    def yolo_dataset_dir(self) -> Path:
        """Scratch train/val split Ultralytics reads from — rebuilt fresh every training run."""
        return self.root / YOLO_DATASET_SUBDIR

    @property
    def input_dir(self) -> Path:
        return self.root / INPUT_SUBDIR

    @property
    def analyzed_dir(self) -> Path:
        return self.root / ANALYZED_SUBDIR

    @property
    def analyzed_approved_dir(self) -> Path:
        return self.analyzed_dir / APPROVED_SUBDIR

    @property
    def analyzed_failed_dir(self) -> Path:
        return self.analyzed_dir / FAILED_SUBDIR

    @property
    def error_dir(self) -> Path:
        """Where the watcher (emil_ml/watcher/) puts a file it couldn't
        process — corrupt input, a not-ready component, any unexpected
        error. Sits beside analyzed_approved_dir/analyzed_failed_dir, not
        inside either: an error is neither a verdict nor a persisted
        inspection."""
        return self.root / ERROR_SUBDIR

    @property
    def knowledge_dir(self) -> Path:
        """Raw .md/.txt knowledge-base documents for this component —
        see core/reporting/knowledge/indexer.py."""
        return self.root / KNOWLEDGE_SUBDIR

    @property
    def archive_dir(self) -> Path:
        """Root of the archive — see archive_dir_for_date() for where an
        individual acknowledged inspection actually lands
        (date-partitioned, not directly under this)."""
        return self.root / ARCHIVE_SUBDIR

    def archive_dir_for_date(self, when: date) -> Path:
        """archive/<year>/<month>/<day>/ for a given date — see
        core/inspections/lifecycle.py, which is what actually moves files
        here. Not part of all_dirs()/create_all(): unlike every other
        subdirectory, which components/all components need from the
        moment they're created, this only comes into existence the first
        time something is actually archived, and there'd be an unbounded
        number of them to pre-create anyway.
        """
        return self.archive_dir / f"{when.year:04d}" / f"{when.month:02d}" / f"{when.day:02d}"

    def all_dirs(self) -> tuple[Path, ...]:
        return (
            self.training_approved_dir,
            self.training_failed_dir,
            self.models_dir,
            self.input_dir,
            self.analyzed_approved_dir,
            self.analyzed_failed_dir,
            self.error_dir,
            self.yolo_images_dir,
            self.yolo_labels_dir,
            self.knowledge_dir,
        )

    def create_all(self) -> None:
        for d in self.all_dirs():
            d.mkdir(parents=True, exist_ok=True)


def for_component(name: str) -> ComponentPaths:
    """Get the ComponentPaths for a given component slug."""
    return ComponentPaths(name=name)
