"""Persistence for the `training_runs` table — the queryable, durable
performance record of every training attempt, independent of whatever a
Streamlit session happened to show once and then lose on the next rerun.

Before this existed, `BaseTrainer.train()` (core/base.py) already returned a
`TrainResult` carrying every metric, confusion matrix, and per-epoch history
in `details` — but nothing kept it: `training/onboard.py`'s
`train_component()` only ever persisted `threshold`/`model_path` to the
`components` row, and `core/search/grid_search.py`'s trials lived only in
memory for the duration of one sweep. This module is the durable home for
that `details` blob, one row per attempt, following the exact
self-contained-schema pattern `core/inspections/store.py` already
established for its own domain-specific table.

`component_name`/`display_name`/`modality`/`model_type` are a snapshot at
the time of the run, not a live foreign key — a row must stay a fully
readable, self-contained history entry even after the component that
produced it is renamed or permanently deleted (in practice it's deleted
right along with the component — see
core/component_deletion.py's permanently_delete_component() — but nothing
here requires the referenced component to still exist).

`settings` is captured because "which settings produced these numbers" is
exactly the kind of context that makes historical metrics useful for
comparison across retrains — reading it back off the *current* component
row would give the wrong answer the moment settings change again.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from emil_ml.config.database import connect, get_connection
from emil_ml.config.settings import DB_PATH, TRAINING_RUNS_TABLE

VALID_SOURCES = ("train", "grid_search")
VALID_STATUSES = ("success", "failed")

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {TRAINING_RUNS_TABLE} (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    component_name        TEXT NOT NULL,
    display_name          TEXT NOT NULL,
    created_at            TEXT NOT NULL DEFAULT (datetime('now')),
    modality              TEXT NOT NULL,
    model_type            TEXT NOT NULL,
    source                TEXT NOT NULL DEFAULT 'train',
    status                TEXT NOT NULL,
    error                 TEXT,
    threshold             REAL,
    model_path            TEXT,
    settings              TEXT NOT NULL DEFAULT '{{}}',
    metrics               TEXT NOT NULL DEFAULT '{{}}',
    details               TEXT NOT NULL DEFAULT '{{}}',
    grid_search_batch_id  TEXT
);
"""

# (column_name, DDL) pairs applied to databases created before that column
# existed — see config/database.py's own _apply_migrations() for the
# pattern this mirrors. `evaluation_dir` is component-root-relative (e.g.
# "evaluation/20260812T101500Z"), same convention model_path already uses
# — see core/base.py's EvaluationResult and training/onboard.py's
# train_component(), the only writer.
_MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("evaluation_dir", f"ALTER TABLE {TRAINING_RUNS_TABLE} ADD COLUMN evaluation_dir TEXT"),
)

# Kept out of the CREATE TABLE's own executescript (and run only after
# _apply_migrations(), not alongside it) — see config/database.py's
# _VERIFIED_INDEX_DDL-after-migrations gotcha this project already hit
# once: an index referencing a column added by a migration must not run in
# the same executescript as a no-op CREATE TABLE IF NOT EXISTS on a
# pre-existing DB, or it executes before the ALTER TABLE that adds the
# column and fails with "no such column".
_INDEX_DDL = f"""
CREATE INDEX IF NOT EXISTS idx_{TRAINING_RUNS_TABLE}_component
    ON {TRAINING_RUNS_TABLE} (component_name, created_at);
"""


@dataclass(frozen=True)
class TrainingRunRecord:
    """A read-only view of one `training_runs` row."""

    id: int
    component_name: str
    display_name: str
    created_at: str
    modality: str
    model_type: str
    source: str
    status: str
    error: str | None
    threshold: float | None
    model_path: str | None
    settings: dict[str, Any]
    metrics: dict[str, Any]
    details: dict[str, Any]
    grid_search_batch_id: str | None
    evaluation_dir: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "TrainingRunRecord":
        return cls(
            id=row["id"],
            component_name=row["component_name"],
            display_name=row["display_name"],
            created_at=row["created_at"],
            modality=row["modality"],
            model_type=row["model_type"],
            source=row["source"],
            status=row["status"],
            error=row["error"],
            threshold=row["threshold"],
            model_path=row["model_path"],
            settings=json.loads(row["settings"] or "{}"),
            metrics=json.loads(row["metrics"] or "{}"),
            details=json.loads(row["details"] or "{}"),
            grid_search_batch_id=row["grid_search_batch_id"],
            evaluation_dir=row["evaluation_dir"],
        )


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    existing_columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({TRAINING_RUNS_TABLE})").fetchall()}
    for column, ddl in _MIGRATIONS:
        if column not in existing_columns:
            conn.execute(ddl)
    conn.executescript(_INDEX_DDL)
    # Must commit here, unconditionally — get()/list_for_component() reach
    # this via config/database.py's get_connection(), which does NOT
    # auto-commit (unlike connect()'s context manager, used by create()/
    # delete_all_for_component()). Without this, a migration triggered by
    # a read call (e.g. opening the Onboard page's Training history for a
    # component before ever calling create() again) silently rolls back
    # when that connection closes, and gets fruitlessly re-attempted on
    # every subsequent read until a write finally commits it for real —
    # confirmed as a real, reproducible bug in the sibling
    # core/cascade/specialists/face/store.py migration (same pattern,
    # same fix); applying the same fix here preemptively rather than
    # waiting to hit it.
    conn.commit()


def create(
    component_name: str,
    *,
    display_name: str,
    modality: str,
    model_type: str,
    status: str,
    source: str = "train",
    error: str | None = None,
    threshold: float | None = None,
    model_path: str | None = None,
    settings: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    details: dict[str, Any] | None = None,
    grid_search_batch_id: str | None = None,
    evaluation_dir: str | None = None,
) -> TrainingRunRecord:
    """Insert one training attempt's record — called once per `BaseTrainer.train()`
    call that actually happens, success or failure alike, so "all performance
    data for every model we train" means every attempt, not just the ones
    that worked.

    `metrics` is kept as its own column, separate from the full `details`
    blob, purely so a history view can render a flat metrics table without
    parsing/filtering the richer `details` JSON (history arrays, confusion
    matrices, ...) every time — same "flat floats vs. nested extras" split
    `TrainResult.details["metrics"]` itself already draws (see core/base.py).
    """
    if source not in VALID_SOURCES:
        raise ValueError(f"Invalid source {source!r}; must be one of {VALID_SOURCES}")
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status {status!r}; must be one of {VALID_STATUSES}")
    with connect(DB_PATH) as conn:
        _ensure_schema(conn)
        cursor = conn.execute(
            f"""INSERT INTO {TRAINING_RUNS_TABLE}
                (component_name, display_name, modality, model_type, source, status, error,
                 threshold, model_path, settings, metrics, details, grid_search_batch_id, evaluation_dir)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                component_name,
                display_name,
                modality,
                model_type,
                source,
                status,
                error,
                threshold,
                model_path,
                json.dumps(settings or {}),
                json.dumps(metrics or {}),
                json.dumps(details or {}),
                grid_search_batch_id,
                evaluation_dir,
            ),
        )
        new_id = cursor.lastrowid
    record = get(new_id)
    assert record is not None
    return record


def get(run_id: int) -> TrainingRunRecord | None:
    conn = get_connection(DB_PATH)
    try:
        _ensure_schema(conn)
        row = conn.execute(f"SELECT * FROM {TRAINING_RUNS_TABLE} WHERE id = ?", (run_id,)).fetchone()
    finally:
        conn.close()
    return TrainingRunRecord.from_row(row) if row else None


def list_for_component(
    component_name: str, *, source: str | None = None, limit: int | None = None
) -> list[TrainingRunRecord]:
    """Every training run recorded for this component, newest first.

    `source=None` (the default) returns both real trainings and grid search
    trials together — pass `source="train"`/`source="grid_search"` to see
    just one kind, e.g. a history view that only cares about actual applied
    trainings, not sweep experiments.
    """
    conn = get_connection(DB_PATH)
    try:
        _ensure_schema(conn)
        query = f"SELECT * FROM {TRAINING_RUNS_TABLE} WHERE component_name = ?"
        params: list[Any] = [component_name]
        if source is not None:
            query += " AND source = ?"
            params.append(source)
        query += " ORDER BY created_at DESC, id DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        rows = conn.execute(query, params).fetchall()
    finally:
        conn.close()
    return [TrainingRunRecord.from_row(r) for r in rows]


def delete_all_for_component(component_name: str) -> int:
    """Permanently remove every training run row for this component in one
    call. Used exclusively by core/component_deletion.py's
    permanently_delete_component() when a whole component is being
    permanently destroyed — same one-shot cascade role
    core/inspections/store.py's delete_all_for_component() plays for that
    table. Idempotent: deleting 0 remaining rows is a safe no-op. Returns
    how many rows were removed.
    """
    with connect(DB_PATH) as conn:
        _ensure_schema(conn)
        cursor = conn.execute(f"DELETE FROM {TRAINING_RUNS_TABLE} WHERE component_name = ?", (component_name,))
        return cursor.rowcount
