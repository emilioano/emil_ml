"""ComponentRegistry: CRUD access over the `components` table."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from emil_ml.config.database import connect, init_db
from emil_ml.config.settings import (
    DB_PATH,
    DEFAULT_APPROVED_HANDLING,
    DEFAULT_AUGMENTATION_STRENGTH,
    DEFAULT_VERIFIED_CORRECTION_POLICY,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CLASSIFIER_BASE_MODEL,
    DEFAULT_CLASSIFIER_POOLING,
    DEFAULT_CLASS_WEIGHT_STRATEGY,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_EPOCHS,
    DEFAULT_FINE_TUNE_EPOCHS,
    DEFAULT_FINE_TUNE_LEARNING_RATE,
    DEFAULT_FINE_TUNE_UNFREEZE_LAYERS,
    DEFAULT_GENERATE_EVALUATION_REPORT,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_INSPECTION_RETENTION_DAYS,
    DEFAULT_ISOLATION_FOREST_CONTAMINATION,
    DEFAULT_ISOLATION_FOREST_MAX_FEATURES,
    DEFAULT_ISOLATION_FOREST_N_ESTIMATORS,
    DEFAULT_ISOLATION_FOREST_STANDARDIZE,
    DEFAULT_LATENT_DIM,
    DEFAULT_LIFECYCLE_STATUS,
    DEFAULT_MACHINE_PARAMETERS,
    DEFAULT_MODALITY,
    DEFAULT_MODEL_TYPE,
    DEFAULT_CASCADE_CATEGORY_SPECIALISTS,
    DEFAULT_CASCADE_STREAM_KAFKA_BOOTSTRAP_SERVERS,
    DEFAULT_CASCADE_STREAM_KAFKA_TOPIC,
    DEFAULT_CASCADE_STREAM_SAMPLE_RATE_SECONDS,
    DEFAULT_COCO_CONFIDENCE_THRESHOLD,
    DEFAULT_PATCHCORE_BACKBONE,
    DEFAULT_PATCHCORE_CORESET_SAMPLING_RATIO,
    DEFAULT_PATCHCORE_NUM_NEIGHBORS,
    DEFAULT_RESNET_CONFIDENCE_THRESHOLD,
    DEFAULT_REPORTING_CLASSES,
    DEFAULT_REPORTING_CONDITION,
    DEFAULT_REPORTING_ENABLED,
    DEFAULT_SCORE_METHOD,
    DEFAULT_THRESHOLD_PERCENTILE,
    DEFAULT_YOLO_AUGMENTATION_STRENGTH,
    DEFAULT_YOLO_CLASS_LOSS_WEIGHT,
    DEFAULT_YOLO_DECISION_RULE,
    DEFAULT_YOLO_LEARNING_RATE,
    DEFAULT_YOLO_MODEL_VARIANT,
    DEFAULT_YOLO_MOSAIC,
    DEFAULT_YOLO_OPTIMIZER,
    VALID_LIFECYCLE_STATUSES,
    VALID_MODALITIES,
    VALID_MODEL_TYPES,
    VALID_STATUSES,
)
from emil_ml.utils.paths import slugify

# Every per-component tunable setting and its default, in one place — the
# "collect each method's parameters in one spot" this registry exists for.
# `create()`/`update_settings()` both key off this dict, so adding a new
# setting (e.g. for a future method) is a one-line change here, not a new
# parameter threaded through several function signatures.
SETTINGS_DEFAULTS: dict[str, Any] = {
    "image_size": DEFAULT_IMAGE_SIZE,
    "epochs": DEFAULT_EPOCHS,
    "latent_dim": DEFAULT_LATENT_DIM,
    "batch_size": DEFAULT_BATCH_SIZE,
    # Shared by every trainable method.
    "early_stopping_patience": DEFAULT_EARLY_STOPPING_PATIENCE,
    # Classifier-only; ignored by other model_types.
    "base_model": DEFAULT_CLASSIFIER_BASE_MODEL,
    "pooling": DEFAULT_CLASSIFIER_POOLING,
    "class_weight_strategy": DEFAULT_CLASS_WEIGHT_STRATEGY,
    "augmentation_strength": DEFAULT_AUGMENTATION_STRENGTH,
    "fine_tune_epochs": DEFAULT_FINE_TUNE_EPOCHS,
    "fine_tune_learning_rate": DEFAULT_FINE_TUNE_LEARNING_RATE,
    "fine_tune_unfreeze_layers": DEFAULT_FINE_TUNE_UNFREEZE_LAYERS,
    # Autoencoder-only; ignored by other model_types.
    "score_method": DEFAULT_SCORE_METHOD,
    "threshold_percentile": DEFAULT_THRESHOLD_PERCENTILE,
    # YOLO-only; ignored by other model_types.
    "yolo_model_variant": DEFAULT_YOLO_MODEL_VARIANT,
    "decision_rule": DEFAULT_YOLO_DECISION_RULE,
    "yolo_mosaic": DEFAULT_YOLO_MOSAIC,
    "yolo_class_loss_weight": DEFAULT_YOLO_CLASS_LOSS_WEIGHT,
    "yolo_augmentation_strength": DEFAULT_YOLO_AUGMENTATION_STRENGTH,
    "yolo_optimizer": DEFAULT_YOLO_OPTIMIZER,
    "yolo_learning_rate": DEFAULT_YOLO_LEARNING_RATE,
    # PatchCore-only; ignored by other model_types.
    "patchcore_backbone": DEFAULT_PATCHCORE_BACKBONE,
    "patchcore_coreset_sampling_ratio": DEFAULT_PATCHCORE_CORESET_SAMPLING_RATIO,
    "patchcore_num_neighbors": DEFAULT_PATCHCORE_NUM_NEIGHBORS,
    # Isolation Forest-only; ignored by other model_types.
    "isolation_forest_n_estimators": DEFAULT_ISOLATION_FOREST_N_ESTIMATORS,
    "isolation_forest_contamination": DEFAULT_ISOLATION_FOREST_CONTAMINATION,
    "isolation_forest_max_features": DEFAULT_ISOLATION_FOREST_MAX_FEATURES,
    "isolation_forest_standardize": DEFAULT_ISOLATION_FOREST_STANDARDIZE,
    # ResNet coarse classifier-only (core/classification/resnet_coarse);
    # ignored by other model_types. See settings.py's own comment for why
    # this model_type has no anomaly_threshold/model_path of its own.
    "resnet_confidence_threshold": DEFAULT_RESNET_CONFIDENCE_THRESHOLD,
    # COCO-YOLO coarse detector-only (core/detection/yolo_coco); ignored by
    # other model_types. Per-detection confidence floor — see settings.py's
    # own comment for why a low-confidence detection is dropped outright
    # rather than mapped to "uncertain" the way the ResNet classifier's
    # single whole-frame guess is.
    "coco_confidence_threshold": DEFAULT_COCO_CONFIDENCE_THRESHOLD,
    # coco_detector-only (core/cascade); ignored by other model_types.
    # Which coarse categories activate a cascade specialist, and which
    # one — see settings.py's own comment and
    # core/cascade/specialist_registry.py's parse/serialize helpers.
    "cascade_category_specialists": DEFAULT_CASCADE_CATEGORY_SPECIALISTS,
    # coco_detector-only (core/cascade live stream, emil_ml/cascade_stream);
    # ignored by other model_types. Kafka connection for a continuous
    # cascade stream — empty means "not configured", not a fallback
    # broker — see settings.py's own comment.
    "cascade_stream_kafka_bootstrap_servers": DEFAULT_CASCADE_STREAM_KAFKA_BOOTSTRAP_SERVERS,
    "cascade_stream_kafka_topic": DEFAULT_CASCADE_STREAM_KAFKA_TOPIC,
    # coco_detector-only. How often (seconds) a continuous source (Kafka or
    # an uploaded video) is actually sampled — see settings.py's own
    # comment and core/cascade/stream_processor.py's should_sample().
    "cascade_stream_sample_rate_seconds": DEFAULT_CASCADE_STREAM_SAMPLE_RATE_SECONDS,
    # Machine context (core/reporting/machine_context) — used by every
    # model_type equally, not tied to any one detection method. A JSON-
    # encoded list of MachineParameterDef (name, unit, normal range, and
    # searchable-state wording per direction); which parameters exist
    # varies per component, so this is one column, not fixed ones — see
    # core/reporting/machine_context/parameters.py.
    "machine_parameters": DEFAULT_MACHINE_PARAMETERS,
    # Reporting opt-in (core/reporting/reporter.py) — off by default. When
    # reporting_enabled is False, reporting_condition/reporting_classes
    # are irrelevant (should_generate_report() short-circuits before
    # ever looking at them).
    "reporting_enabled": DEFAULT_REPORTING_ENABLED,
    "reporting_condition": DEFAULT_REPORTING_CONDITION,
    "reporting_classes": DEFAULT_REPORTING_CLASSES,
    # core/inspections — how long an archived inspection's files are kept
    # before a (not yet built) cleanup job could delete them. The setting
    # exists now so it's already in place when that job is; nothing reads
    # it yet.
    "inspection_retention_days": DEFAULT_INSPECTION_RETENTION_DAYS,
    # How an approved verdict is handled — see settings.py's
    # VALID_APPROVED_HANDLING_MODES for the three modes and the reasoning
    # (a false negative is a real, permanent record regardless of mode;
    # they only differ in default visibility and acknowledgement).
    "approved_handling": DEFAULT_APPROVED_HANDLING,
    # Whether/how a human-verified correction gets copied into this
    # component's actual training/ data — see settings.py's
    # VALID_VERIFIED_CORRECTION_POLICIES for the three modes (off/
    # manual_review/automatic) and the reasoning.
    "verified_correction_policy": DEFAULT_VERIFIED_CORRECTION_POLICY,
    # Whether train_component() generates+saves a model_type-aware
    # evaluation report (metrics + plots) after every successful training
    # run — see settings.py's own comment and core/base.py's
    # BaseTrainer.evaluate().
    "generate_evaluation_report": DEFAULT_GENERATE_EVALUATION_REPORT,
}


@dataclass(frozen=True)
class Component:
    """A read-only view of a `components` row."""

    id: int
    name: str
    display_name: str
    created_at: str
    image_size: int
    epochs: int
    latent_dim: int
    batch_size: int
    early_stopping_patience: int
    modality: str
    model_type: str
    base_model: str
    pooling: str
    class_weight_strategy: str
    augmentation_strength: float
    fine_tune_epochs: int
    fine_tune_learning_rate: float
    fine_tune_unfreeze_layers: int
    score_method: str
    threshold_percentile: float
    yolo_model_variant: str
    decision_rule: str
    yolo_mosaic: float
    yolo_class_loss_weight: float
    yolo_augmentation_strength: float
    yolo_optimizer: str
    yolo_learning_rate: float
    patchcore_backbone: str
    patchcore_coreset_sampling_ratio: float
    patchcore_num_neighbors: int
    isolation_forest_n_estimators: int
    isolation_forest_contamination: str
    isolation_forest_max_features: float
    isolation_forest_standardize: int
    resnet_confidence_threshold: float
    coco_confidence_threshold: float
    cascade_category_specialists: str
    cascade_stream_kafka_bootstrap_servers: str
    cascade_stream_kafka_topic: str
    cascade_stream_sample_rate_seconds: float
    machine_parameters: str
    reporting_enabled: int
    reporting_condition: str
    reporting_classes: str
    inspection_retention_days: int
    approved_handling: str
    verified_correction_policy: str
    generate_evaluation_report: int
    lifecycle_status: str
    deleted_at: str | None
    anomaly_threshold: float | None
    model_path: str | None
    status: str

    @classmethod
    def from_row(cls, row: Any) -> "Component":
        return cls(**{k: row[k] for k in row.keys()})


class ComponentRegistry:
    """CRUD operations over the SQLite `components` table."""

    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        self.db_path = db_path
        init_db(self.db_path)

    def create(
        self,
        display_name: str,
        *,
        modality: str = DEFAULT_MODALITY,
        model_type: str = DEFAULT_MODEL_TYPE,
        **settings: Any,
    ) -> Component:
        """Insert a new component with status='created' and return it.

        Any setting from SETTINGS_DEFAULTS can be overridden by keyword;
        unspecified ones use their default.
        """
        if modality not in VALID_MODALITIES:
            raise ValueError(f"Invalid modality {modality!r}; must be one of {VALID_MODALITIES}")
        if model_type not in VALID_MODEL_TYPES:
            raise ValueError(f"Invalid model_type {model_type!r}; must be one of {VALID_MODEL_TYPES}")
        unknown = set(settings) - set(SETTINGS_DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown setting(s): {sorted(unknown)}")

        merged = {**SETTINGS_DEFAULTS, **settings}
        name = slugify(display_name)
        columns = ["name", "display_name", "modality", "model_type", *merged.keys()]
        values = [name, display_name, modality, model_type, *merged.values()]
        placeholders = ", ".join(["?"] * len(values))
        with connect(self.db_path) as conn:
            conn.execute(
                f"INSERT INTO components ({', '.join(columns)}, status) "
                f"VALUES ({placeholders}, 'created')",
                values,
            )
        component = self.get(name)
        assert component is not None
        return component

    def get(self, name: str) -> Component | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM components WHERE name = ?", (name,)
            ).fetchone()
        return Component.from_row(row) if row else None

    def get_by_id(self, component_id: int) -> Component | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM components WHERE id = ?", (component_id,)
            ).fetchone()
        return Component.from_row(row) if row else None

    def list_all(self, *, include_deleted: bool = False) -> list[Component]:
        """Every component EXCEPT soft-deleted ones by default — this is
        the right default for almost every existing call site (Onboard
        page pickers, dashboard counts, filter dropdowns): a soft-deleted
        component belongs in the trash view only (see list_deleted()),
        never mixed into a normal "pick a component" list. Active AND
        inactive components are both included — inactive only means
        "excluded from active-use iteration" (the watcher, RAG indexing,
        the Inspect page's picker — see list_active()), not "hidden from
        management", since you still need to find and reactivate/edit/
        retrain it somewhere.

        `include_deleted=True` is for internal use (e.g. component_deletion.py
        double-checking state) — there's no normal UI reason to want
        active+inactive+deleted all mixed together; use list_deleted()
        for the trash view instead.
        """
        with connect(self.db_path) as conn:
            query = "SELECT * FROM components"
            if not include_deleted:
                query += " WHERE lifecycle_status != 'deleted'"
            query += " ORDER BY created_at DESC"
            rows = conn.execute(query).fetchall()
        return [Component.from_row(row) for row in rows]

    def list_active(self) -> list[Component]:
        """Only lifecycle_status='active' components — what every
        registry-driven ACTIVE-USE iteration (the watcher's input/ scan,
        RAG's index_all(), the Inspect page's component picker) should
        use instead of list_all(), so a deactivated or soft-deleted
        component is never watched, reindexed, or offered for inspection.
        """
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM components WHERE lifecycle_status = 'active' ORDER BY display_name"
            ).fetchall()
        return [Component.from_row(row) for row in rows]

    def list_deleted(self) -> list[Component]:
        """Only soft-deleted components, newest-deleted first — the
        trash view's data source. Nothing else in the app should ever
        show these mixed in with normal components."""
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM components WHERE lifecycle_status = 'deleted' ORDER BY deleted_at DESC"
            ).fetchall()
        return [Component.from_row(row) for row in rows]

    def list_ready(self) -> list[Component]:
        """Trained (status='ready') AND active (lifecycle_status='active')
        — the Inspect page's component picker uses this, so a
        deactivated or soft-deleted component is never offered for
        inspection even if it was fully trained before being paused.
        """
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM components WHERE status = 'ready' AND lifecycle_status = 'active' "
                "ORDER BY display_name"
            ).fetchall()
        return [Component.from_row(row) for row in rows]

    def update_status(self, name: str, status: str) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status {status!r}; must be one of {VALID_STATUSES}")
        with connect(self.db_path) as conn:
            conn.execute(
                "UPDATE components SET status = ? WHERE name = ?", (status, name)
            )

    def update_training_result(
        self,
        name: str,
        *,
        anomaly_threshold: float | None,
        model_path: str | None,
        status: str = "ready",
    ) -> None:
        """Record the outcome of a training run."""
        with connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE components
                SET anomaly_threshold = ?, model_path = ?, status = ?
                WHERE name = ?
                """,
                (anomaly_threshold, model_path, status, name),
            )

    def update_threshold(self, name: str, threshold: float) -> None:
        """Persist a new decision threshold without retraining.

        The threshold (reconstruction-error cutoff, classifier probability,
        or YOLO confidence, depending on model_type) is just a cutoff applied
        to scores the model already produces — changing it doesn't require
        re-fitting anything, unlike every other setting in SETTINGS_DEFAULTS.
        """
        if self.get(name) is None:
            raise KeyError(f"No component named {name!r}")
        with connect(self.db_path) as conn:
            conn.execute(
                "UPDATE components SET anomaly_threshold = ? WHERE name = ?", (threshold, name)
            )

    def rename_display_name(self, name: str, new_display_name: str) -> None:
        """Change only the human-readable label shown throughout the UI.

        Deliberately NOT the same as renaming `name` (the slug): `name` is
        the stable identifier everything else keys off — the on-disk
        folder (utils/paths.py's ComponentPaths), ChromaDB's
        `component_type` metadata, and the TEXT component_name columns in
        `inspections`/`machine_readings` (core/inspections/store.py,
        core/reporting/machine_context/source.py) all reference it, none
        of them via a stable numeric id. Changing `display_name` alone
        touches none of that — it's a pure label edit, safe to do anytime
        without affecting a trained model, indexed documentation, or
        inspection history.
        """
        if not new_display_name.strip():
            raise ValueError("display_name cannot be empty")
        if self.get(name) is None:
            raise KeyError(f"No component named {name!r}")
        with connect(self.db_path) as conn:
            conn.execute(
                "UPDATE components SET display_name = ? WHERE name = ?", (new_display_name.strip(), name)
            )

    def update_settings(self, name: str, **settings: Any) -> None:
        """Update any subset of SETTINGS_DEFAULTS, leaving unspecified fields unchanged."""
        if not settings:
            return
        unknown = set(settings) - set(SETTINGS_DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown setting(s): {sorted(unknown)}")
        if self.get(name) is None:
            raise KeyError(f"No component named {name!r}")

        assignments = ", ".join(f"{column} = ?" for column in settings)
        with connect(self.db_path) as conn:
            conn.execute(
                f"UPDATE components SET {assignments} WHERE name = ?",
                [*settings.values(), name],
            )

    def delete(self, name: str) -> None:
        """Remove ONLY the registry row — a low-level primitive, not the
        "delete a component" action a user ever triggers directly.

        Does not touch this component's files, inspections,
        machine_readings, or ChromaDB chunks — a caller that wants those
        cleaned up too needs core/component_deletion.py's
        permanently_delete_component(), which calls this as its own last
        step (after everything else, so re-running it is always safe —
        see that function's own docstring). Used directly by test/verify
        scripts that need a disposable component fully forgotten by the
        registry without caring about the rest of its footprint.
        """
        with connect(self.db_path) as conn:
            conn.execute("DELETE FROM components WHERE name = ?", (name,))

    def deactivate(self, name: str) -> None:
        """active -> inactive. Reversible, zero data impact: models,
        training data, inspection history, knowledge docs, and ChromaDB
        chunks are all untouched — this only excludes the component from
        active-use iteration (list_active(), used by the watcher's
        input/ scan, RAG's index_all(), and the Inspect page's component
        picker). A no-op if the component is already 'deleted' (soft-
        delete is a stronger state; deactivating a trashed component
        makes no sense — restore() it first).
        """
        if self.get(name) is None:
            raise KeyError(f"No component named {name!r}")
        with connect(self.db_path) as conn:
            conn.execute(
                "UPDATE components SET lifecycle_status = 'inactive' "
                "WHERE name = ? AND lifecycle_status = 'active'",
                (name,),
            )

    def reactivate(self, name: str) -> None:
        """inactive -> active. A plain flag flip back — nothing to
        restore, since deactivate() never removed anything."""
        if self.get(name) is None:
            raise KeyError(f"No component named {name!r}")
        with connect(self.db_path) as conn:
            conn.execute(
                "UPDATE components SET lifecycle_status = 'active' "
                "WHERE name = ? AND lifecycle_status = 'inactive'",
                (name,),
            )

    def soft_delete(self, name: str) -> None:
        """active/inactive -> deleted. Moves the component to the trash:
        excluded from active use (same as inactive) AND hidden from
        normal management views (list_all()'s default), reachable only
        via list_deleted(). All data untouched and fully reversible via
        restore() — soft-deleting never removes a single file, row, or
        chunk. Only core/component_deletion.py's
        permanently_delete_component() actually destroys anything, and
        only for a component already in this state.
        """
        if self.get(name) is None:
            raise KeyError(f"No component named {name!r}")
        with connect(self.db_path) as conn:
            conn.execute(
                "UPDATE components SET lifecycle_status = 'deleted', deleted_at = datetime('now') "
                "WHERE name = ? AND lifecycle_status != 'deleted'",
                (name,),
            )

    def restore(self, name: str) -> None:
        """deleted -> active. Undoes soft_delete() completely — there's
        nothing partial to restore, since nothing was ever removed, just
        hidden. Clears deleted_at so a subsequent soft_delete() gets a
        fresh timestamp rather than an old one lingering.
        """
        if self.get(name) is None:
            raise KeyError(f"No component named {name!r}")
        with connect(self.db_path) as conn:
            conn.execute(
                "UPDATE components SET lifecycle_status = 'active', deleted_at = NULL "
                "WHERE name = ? AND lifecycle_status = 'deleted'",
                (name,),
            )
