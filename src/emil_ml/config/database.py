"""SQLite connection and schema management for EMIL."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from emil_ml.config.settings import (
    DB_PATH,
    DEFAULT_APPROVED_HANDLING,
    DEFAULT_AUGMENTATION_STRENGTH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CASCADE_CATEGORY_SPECIALISTS,
    DEFAULT_CASCADE_STREAM_KAFKA_BOOTSTRAP_SERVERS,
    DEFAULT_CASCADE_STREAM_KAFKA_TOPIC,
    DEFAULT_CASCADE_STREAM_SAMPLE_RATE_SECONDS,
    DEFAULT_CLASSIFIER_BASE_MODEL,
    DEFAULT_CLASSIFIER_POOLING,
    DEFAULT_CLASS_WEIGHT_STRATEGY,
    DEFAULT_COCO_CONFIDENCE_THRESHOLD,
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
    DEFAULT_PATCHCORE_BACKBONE,
    DEFAULT_PATCHCORE_CORESET_SAMPLING_RATIO,
    DEFAULT_PATCHCORE_NUM_NEIGHBORS,
    DEFAULT_RESNET_CONFIDENCE_THRESHOLD,
    DEFAULT_REPORTING_CLASSES,
    DEFAULT_REPORTING_CONDITION,
    DEFAULT_REPORTING_ENABLED,
    DEFAULT_SCORE_METHOD,
    DEFAULT_THRESHOLD_PERCENTILE,
    DEFAULT_VERIFIED_CORRECTION_POLICY,
    DEFAULT_YOLO_AUGMENTATION_STRENGTH,
    DEFAULT_YOLO_CLASS_LOSS_WEIGHT,
    DEFAULT_YOLO_DECISION_RULE,
    DEFAULT_YOLO_LEARNING_RATE,
    DEFAULT_YOLO_MODEL_VARIANT,
    DEFAULT_YOLO_MOSAIC,
    DEFAULT_YOLO_OPTIMIZER,
)

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS components (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    name                     TEXT NOT NULL UNIQUE,
    display_name             TEXT NOT NULL,
    created_at               TEXT NOT NULL DEFAULT (datetime('now')),
    image_size               INTEGER NOT NULL DEFAULT {DEFAULT_IMAGE_SIZE},
    epochs                   INTEGER NOT NULL DEFAULT {DEFAULT_EPOCHS},
    latent_dim                INTEGER NOT NULL DEFAULT {DEFAULT_LATENT_DIM},
    batch_size               INTEGER NOT NULL DEFAULT {DEFAULT_BATCH_SIZE},
    modality                 TEXT NOT NULL DEFAULT '{DEFAULT_MODALITY}',
    model_type               TEXT NOT NULL DEFAULT '{DEFAULT_MODEL_TYPE}',
    base_model               TEXT NOT NULL DEFAULT '{DEFAULT_CLASSIFIER_BASE_MODEL}',
    pooling                  TEXT NOT NULL DEFAULT '{DEFAULT_CLASSIFIER_POOLING}',
    score_method             TEXT NOT NULL DEFAULT '{DEFAULT_SCORE_METHOD}',
    threshold_percentile     REAL NOT NULL DEFAULT {DEFAULT_THRESHOLD_PERCENTILE},
    class_weight_strategy    TEXT NOT NULL DEFAULT '{DEFAULT_CLASS_WEIGHT_STRATEGY}',
    augmentation_strength    REAL NOT NULL DEFAULT {DEFAULT_AUGMENTATION_STRENGTH},
    fine_tune_epochs         INTEGER NOT NULL DEFAULT {DEFAULT_FINE_TUNE_EPOCHS},
    fine_tune_learning_rate  REAL NOT NULL DEFAULT {DEFAULT_FINE_TUNE_LEARNING_RATE},
    fine_tune_unfreeze_layers INTEGER NOT NULL DEFAULT {DEFAULT_FINE_TUNE_UNFREEZE_LAYERS},
    early_stopping_patience  INTEGER NOT NULL DEFAULT {DEFAULT_EARLY_STOPPING_PATIENCE},
    yolo_model_variant       TEXT NOT NULL DEFAULT '{DEFAULT_YOLO_MODEL_VARIANT}',
    decision_rule            TEXT NOT NULL DEFAULT '{DEFAULT_YOLO_DECISION_RULE}',
    yolo_mosaic              REAL NOT NULL DEFAULT {DEFAULT_YOLO_MOSAIC},
    yolo_class_loss_weight   REAL NOT NULL DEFAULT {DEFAULT_YOLO_CLASS_LOSS_WEIGHT},
    yolo_augmentation_strength REAL NOT NULL DEFAULT {DEFAULT_YOLO_AUGMENTATION_STRENGTH},
    yolo_optimizer           TEXT NOT NULL DEFAULT '{DEFAULT_YOLO_OPTIMIZER}',
    yolo_learning_rate       REAL NOT NULL DEFAULT {DEFAULT_YOLO_LEARNING_RATE},
    patchcore_backbone       TEXT NOT NULL DEFAULT '{DEFAULT_PATCHCORE_BACKBONE}',
    patchcore_coreset_sampling_ratio REAL NOT NULL DEFAULT {DEFAULT_PATCHCORE_CORESET_SAMPLING_RATIO},
    patchcore_num_neighbors  INTEGER NOT NULL DEFAULT {DEFAULT_PATCHCORE_NUM_NEIGHBORS},
    isolation_forest_n_estimators INTEGER NOT NULL DEFAULT {DEFAULT_ISOLATION_FOREST_N_ESTIMATORS},
    isolation_forest_contamination TEXT NOT NULL DEFAULT '{DEFAULT_ISOLATION_FOREST_CONTAMINATION}',
    isolation_forest_max_features REAL NOT NULL DEFAULT {DEFAULT_ISOLATION_FOREST_MAX_FEATURES},
    isolation_forest_standardize INTEGER NOT NULL DEFAULT {int(DEFAULT_ISOLATION_FOREST_STANDARDIZE)},
    resnet_confidence_threshold REAL NOT NULL DEFAULT {DEFAULT_RESNET_CONFIDENCE_THRESHOLD},
    coco_confidence_threshold REAL NOT NULL DEFAULT {DEFAULT_COCO_CONFIDENCE_THRESHOLD},
    cascade_category_specialists TEXT NOT NULL DEFAULT '{DEFAULT_CASCADE_CATEGORY_SPECIALISTS}',
    cascade_stream_kafka_bootstrap_servers TEXT NOT NULL DEFAULT '{DEFAULT_CASCADE_STREAM_KAFKA_BOOTSTRAP_SERVERS}',
    cascade_stream_kafka_topic TEXT NOT NULL DEFAULT '{DEFAULT_CASCADE_STREAM_KAFKA_TOPIC}',
    cascade_stream_sample_rate_seconds REAL NOT NULL DEFAULT {DEFAULT_CASCADE_STREAM_SAMPLE_RATE_SECONDS},
    machine_parameters       TEXT NOT NULL DEFAULT '{DEFAULT_MACHINE_PARAMETERS}',
    reporting_enabled        INTEGER NOT NULL DEFAULT {int(DEFAULT_REPORTING_ENABLED)},
    reporting_condition      TEXT NOT NULL DEFAULT '{DEFAULT_REPORTING_CONDITION}',
    reporting_classes        TEXT NOT NULL DEFAULT '{DEFAULT_REPORTING_CLASSES}',
    inspection_retention_days INTEGER NOT NULL DEFAULT {DEFAULT_INSPECTION_RETENTION_DAYS},
    approved_handling        TEXT NOT NULL DEFAULT '{DEFAULT_APPROVED_HANDLING}',
    verified_correction_policy TEXT NOT NULL DEFAULT '{DEFAULT_VERIFIED_CORRECTION_POLICY}',
    generate_evaluation_report INTEGER NOT NULL DEFAULT {int(DEFAULT_GENERATE_EVALUATION_REPORT)},
    lifecycle_status         TEXT NOT NULL DEFAULT '{DEFAULT_LIFECYCLE_STATUS}',
    deleted_at               TEXT,
    anomaly_threshold        REAL,
    model_path               TEXT,
    status                   TEXT NOT NULL DEFAULT 'created'
);
"""

# (column_name, DDL) pairs applied to databases created before that column existed.
_MIGRATIONS: tuple[tuple[str, str], ...] = (
    (
        "model_type",
        f"ALTER TABLE components ADD COLUMN model_type TEXT NOT NULL DEFAULT '{DEFAULT_MODEL_TYPE}'",
    ),
    (
        "modality",
        f"ALTER TABLE components ADD COLUMN modality TEXT NOT NULL DEFAULT '{DEFAULT_MODALITY}'",
    ),
    (
        "base_model",
        f"ALTER TABLE components ADD COLUMN base_model TEXT NOT NULL DEFAULT '{DEFAULT_CLASSIFIER_BASE_MODEL}'",
    ),
    (
        "pooling",
        f"ALTER TABLE components ADD COLUMN pooling TEXT NOT NULL DEFAULT '{DEFAULT_CLASSIFIER_POOLING}'",
    ),
    (
        "score_method",
        f"ALTER TABLE components ADD COLUMN score_method TEXT NOT NULL DEFAULT '{DEFAULT_SCORE_METHOD}'",
    ),
    (
        "threshold_percentile",
        f"ALTER TABLE components ADD COLUMN threshold_percentile REAL NOT NULL DEFAULT {DEFAULT_THRESHOLD_PERCENTILE}",
    ),
    (
        "class_weight_strategy",
        f"ALTER TABLE components ADD COLUMN class_weight_strategy TEXT NOT NULL DEFAULT '{DEFAULT_CLASS_WEIGHT_STRATEGY}'",
    ),
    (
        "augmentation_strength",
        f"ALTER TABLE components ADD COLUMN augmentation_strength REAL NOT NULL DEFAULT {DEFAULT_AUGMENTATION_STRENGTH}",
    ),
    (
        "fine_tune_epochs",
        f"ALTER TABLE components ADD COLUMN fine_tune_epochs INTEGER NOT NULL DEFAULT {DEFAULT_FINE_TUNE_EPOCHS}",
    ),
    (
        "fine_tune_learning_rate",
        "ALTER TABLE components ADD COLUMN fine_tune_learning_rate REAL NOT NULL DEFAULT "
        f"{DEFAULT_FINE_TUNE_LEARNING_RATE}",
    ),
    (
        "fine_tune_unfreeze_layers",
        "ALTER TABLE components ADD COLUMN fine_tune_unfreeze_layers INTEGER NOT NULL DEFAULT "
        f"{DEFAULT_FINE_TUNE_UNFREEZE_LAYERS}",
    ),
    (
        "early_stopping_patience",
        "ALTER TABLE components ADD COLUMN early_stopping_patience INTEGER NOT NULL DEFAULT "
        f"{DEFAULT_EARLY_STOPPING_PATIENCE}",
    ),
    (
        "yolo_model_variant",
        f"ALTER TABLE components ADD COLUMN yolo_model_variant TEXT NOT NULL DEFAULT '{DEFAULT_YOLO_MODEL_VARIANT}'",
    ),
    (
        "decision_rule",
        f"ALTER TABLE components ADD COLUMN decision_rule TEXT NOT NULL DEFAULT '{DEFAULT_YOLO_DECISION_RULE}'",
    ),
    (
        "yolo_mosaic",
        f"ALTER TABLE components ADD COLUMN yolo_mosaic REAL NOT NULL DEFAULT {DEFAULT_YOLO_MOSAIC}",
    ),
    (
        "yolo_class_loss_weight",
        "ALTER TABLE components ADD COLUMN yolo_class_loss_weight REAL NOT NULL DEFAULT "
        f"{DEFAULT_YOLO_CLASS_LOSS_WEIGHT}",
    ),
    (
        "yolo_augmentation_strength",
        "ALTER TABLE components ADD COLUMN yolo_augmentation_strength REAL NOT NULL DEFAULT "
        f"{DEFAULT_YOLO_AUGMENTATION_STRENGTH}",
    ),
    (
        "yolo_optimizer",
        f"ALTER TABLE components ADD COLUMN yolo_optimizer TEXT NOT NULL DEFAULT '{DEFAULT_YOLO_OPTIMIZER}'",
    ),
    (
        "yolo_learning_rate",
        "ALTER TABLE components ADD COLUMN yolo_learning_rate REAL NOT NULL DEFAULT "
        f"{DEFAULT_YOLO_LEARNING_RATE}",
    ),
    (
        "patchcore_backbone",
        f"ALTER TABLE components ADD COLUMN patchcore_backbone TEXT NOT NULL DEFAULT '{DEFAULT_PATCHCORE_BACKBONE}'",
    ),
    (
        "patchcore_coreset_sampling_ratio",
        "ALTER TABLE components ADD COLUMN patchcore_coreset_sampling_ratio REAL NOT NULL DEFAULT "
        f"{DEFAULT_PATCHCORE_CORESET_SAMPLING_RATIO}",
    ),
    (
        "patchcore_num_neighbors",
        "ALTER TABLE components ADD COLUMN patchcore_num_neighbors INTEGER NOT NULL DEFAULT "
        f"{DEFAULT_PATCHCORE_NUM_NEIGHBORS}",
    ),
    (
        "isolation_forest_n_estimators",
        "ALTER TABLE components ADD COLUMN isolation_forest_n_estimators INTEGER NOT NULL DEFAULT "
        f"{DEFAULT_ISOLATION_FOREST_N_ESTIMATORS}",
    ),
    (
        "isolation_forest_contamination",
        "ALTER TABLE components ADD COLUMN isolation_forest_contamination TEXT NOT NULL DEFAULT "
        f"'{DEFAULT_ISOLATION_FOREST_CONTAMINATION}'",
    ),
    (
        "isolation_forest_max_features",
        "ALTER TABLE components ADD COLUMN isolation_forest_max_features REAL NOT NULL DEFAULT "
        f"{DEFAULT_ISOLATION_FOREST_MAX_FEATURES}",
    ),
    (
        "isolation_forest_standardize",
        "ALTER TABLE components ADD COLUMN isolation_forest_standardize INTEGER NOT NULL DEFAULT "
        f"{int(DEFAULT_ISOLATION_FOREST_STANDARDIZE)}",
    ),
    (
        "machine_parameters",
        f"ALTER TABLE components ADD COLUMN machine_parameters TEXT NOT NULL DEFAULT "
        f"'{DEFAULT_MACHINE_PARAMETERS}'",
    ),
    (
        "reporting_enabled",
        "ALTER TABLE components ADD COLUMN reporting_enabled INTEGER NOT NULL DEFAULT "
        f"{int(DEFAULT_REPORTING_ENABLED)}",
    ),
    (
        "reporting_condition",
        "ALTER TABLE components ADD COLUMN reporting_condition TEXT NOT NULL DEFAULT "
        f"'{DEFAULT_REPORTING_CONDITION}'",
    ),
    (
        "reporting_classes",
        "ALTER TABLE components ADD COLUMN reporting_classes TEXT NOT NULL DEFAULT "
        f"'{DEFAULT_REPORTING_CLASSES}'",
    ),
    (
        "inspection_retention_days",
        "ALTER TABLE components ADD COLUMN inspection_retention_days INTEGER NOT NULL DEFAULT "
        f"{DEFAULT_INSPECTION_RETENTION_DAYS}",
    ),
    (
        "approved_handling",
        "ALTER TABLE components ADD COLUMN approved_handling TEXT NOT NULL DEFAULT "
        f"'{DEFAULT_APPROVED_HANDLING}'",
    ),
    (
        "verified_correction_policy",
        "ALTER TABLE components ADD COLUMN verified_correction_policy TEXT NOT NULL DEFAULT "
        f"'{DEFAULT_VERIFIED_CORRECTION_POLICY}'",
    ),
    (
        "lifecycle_status",
        "ALTER TABLE components ADD COLUMN lifecycle_status TEXT NOT NULL DEFAULT "
        f"'{DEFAULT_LIFECYCLE_STATUS}'",
    ),
    ("deleted_at", "ALTER TABLE components ADD COLUMN deleted_at TEXT"),
    (
        "resnet_confidence_threshold",
        "ALTER TABLE components ADD COLUMN resnet_confidence_threshold REAL NOT NULL DEFAULT "
        f"{DEFAULT_RESNET_CONFIDENCE_THRESHOLD}",
    ),
    (
        "coco_confidence_threshold",
        "ALTER TABLE components ADD COLUMN coco_confidence_threshold REAL NOT NULL DEFAULT "
        f"{DEFAULT_COCO_CONFIDENCE_THRESHOLD}",
    ),
    (
        "generate_evaluation_report",
        "ALTER TABLE components ADD COLUMN generate_evaluation_report INTEGER NOT NULL DEFAULT "
        f"{int(DEFAULT_GENERATE_EVALUATION_REPORT)}",
    ),
    (
        "cascade_category_specialists",
        "ALTER TABLE components ADD COLUMN cascade_category_specialists TEXT NOT NULL DEFAULT "
        f"'{DEFAULT_CASCADE_CATEGORY_SPECIALISTS}'",
    ),
    (
        "cascade_stream_kafka_bootstrap_servers",
        "ALTER TABLE components ADD COLUMN cascade_stream_kafka_bootstrap_servers TEXT NOT NULL DEFAULT "
        f"'{DEFAULT_CASCADE_STREAM_KAFKA_BOOTSTRAP_SERVERS}'",
    ),
    (
        "cascade_stream_kafka_topic",
        "ALTER TABLE components ADD COLUMN cascade_stream_kafka_topic TEXT NOT NULL DEFAULT "
        f"'{DEFAULT_CASCADE_STREAM_KAFKA_TOPIC}'",
    ),
    (
        "cascade_stream_sample_rate_seconds",
        "ALTER TABLE components ADD COLUMN cascade_stream_sample_rate_seconds REAL NOT NULL DEFAULT "
        f"{DEFAULT_CASCADE_STREAM_SAMPLE_RATE_SECONDS}",
    ),
)

# Columns from an earlier machine-context design (fixed per-parameter
# normal-range columns) superseded by the generic `machine_parameters`
# JSON column above — see core/reporting/machine_context/parameters.py.
# Dropped outright rather than left as dead columns: this project has no
# external users/compatibility burden, and leaving ten unused
# temperature_normal_min-style columns next to machine_parameters would
# just be confusing. Requires SQLite >= 3.35 (DROP COLUMN); this project
# targets a bundled sqlite3 well past that.
_DROPPED_COLUMNS: tuple[str, ...] = (
    "temperature_normal_min",
    "temperature_normal_max",
    "speed_normal_min",
    "speed_normal_max",
    "pressure_normal_min",
    "pressure_normal_max",
    "vibration_normal_min",
    "vibration_normal_max",
    "hours_since_service_normal_min",
    "hours_since_service_normal_max",
)


def _apply_migrations(conn: sqlite3.Connection) -> None:
    existing_columns = {row["name"] for row in conn.execute("PRAGMA table_info(components)").fetchall()}
    for column, ddl in _MIGRATIONS:
        if column not in existing_columns:
            conn.execute(ddl)
    for column in _DROPPED_COLUMNS:
        if column in existing_columns:
            conn.execute(f"ALTER TABLE components DROP COLUMN {column}")


def get_connection(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with row access by column name and FKs enabled."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: Path | str = DB_PATH) -> None:
    """Create the database schema if needed, and migrate older databases in place."""
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)
        _apply_migrations(conn)
        conn.commit()


@contextmanager
def connect(db_path: Path | str = DB_PATH) -> Iterator[sqlite3.Connection]:
    """Context-managed connection that commits on success and closes always."""
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
