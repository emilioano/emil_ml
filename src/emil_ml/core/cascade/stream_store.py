"""Persistence for the `cascade_stream_runs`/`cascade_stream_results` tables
— the durable record of continuous cascade operation (a Kafka consumer
process's lifetime, one uploaded-video pass, or one uploaded still image),
independent of any single Streamlit session. Same self-contained-schema pattern
core/training_runs/store.py and core/inspections/store.py already
established for their own domain-specific tables — this is telemetry/
results data, not a per-component setting, so it doesn't go through
config/database.py's SCHEMA (see settings.py's own comment on
CASCADE_STREAM_RUNS_TABLE/CASCADE_STREAM_RESULTS_TABLE).

Two tables, not one:
- `cascade_stream_runs`: one row per "run" — a Kafka consumer process's
  start-to-stop lifetime, one video-file processing pass, or one uploaded
  still image. Carries the
  liveness heartbeat (see emil_ml.cascade_stream.service) a Streamlit page
  can poll to tell "actually running" from "stopped" or "crashed without
  saying so" — see CASCADE_STREAM_HEARTBEAT_STALE_SECONDS.
- `cascade_stream_results`: one row per actually-processed (post-throttle)
  frame, FK'd to its run. Doesn't fit core/inspections/store.py's
  InspectionRecord shape at all — that table hard-assumes exactly one
  verdict/score per row, singular NOT NULL columns; a cascade frame
  produces zero-to-many DetectedObjects, each with its own optional
  specialist/policy outcome, which is exactly why this is a new table
  rather than a shoehorned column there.

`component_name` is captured on both tables as a snapshot, same convention
training_runs/store.py already documents: a row stays a self-contained
history entry even after the component is renamed or deleted (in practice
deleted right along with it — see core/component_deletion.py).
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from emil_ml.config.database import connect, get_connection
from emil_ml.config.settings import CASCADE_STREAM_RESULTS_TABLE, CASCADE_STREAM_RUNS_TABLE, DB_PATH

if TYPE_CHECKING:
    from emil_ml.core.cascade.pipeline import DetectedObject

VALID_SOURCES = ("kafka", "video", "image")
VALID_RUN_STATUSES = ("running", "stopped", "crashed", "completed")

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {CASCADE_STREAM_RUNS_TABLE} (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    component_name    TEXT NOT NULL,
    source            TEXT NOT NULL,
    source_detail     TEXT NOT NULL DEFAULT '',
    started_at        TEXT NOT NULL DEFAULT (datetime('now')),
    last_heartbeat_at TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at       TEXT,
    status            TEXT NOT NULL DEFAULT 'running',
    frames_seen       INTEGER NOT NULL DEFAULT 0,
    frames_processed  INTEGER NOT NULL DEFAULT 0,
    last_error        TEXT
);

CREATE TABLE IF NOT EXISTS {CASCADE_STREAM_RESULTS_TABLE} (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL,
    component_name  TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    source          TEXT NOT NULL,
    frame_ref       TEXT NOT NULL,
    object_count    INTEGER NOT NULL DEFAULT 0,
    objects_json    TEXT NOT NULL DEFAULT '[]',
    thumbnail_path  TEXT
);
"""

# (column_name, DDL) pairs applied to databases created before that column
# existed — see config/database.py's own _apply_migrations() for the
# pattern this mirrors. Empty today (both tables are new); kept for the
# same forward-compatibility reason training_runs/store.py keeps its own
# (initially-empty-then-grown) _MIGRATIONS tuple.
_MIGRATIONS: tuple[tuple[str, str], ...] = ()

# Kept out of the CREATE TABLE executescript, run only after
# _apply_migrations() — see config/database.py's own documented gotcha
# (also called out in training_runs/store.py): an index referencing a
# migration-added column must not run in the same executescript as a no-op
# CREATE TABLE IF NOT EXISTS on a pre-existing DB, or it can execute before
# the ALTER TABLE that adds the column.
_INDEX_DDL = f"""
CREATE INDEX IF NOT EXISTS idx_{CASCADE_STREAM_RUNS_TABLE}_component
    ON {CASCADE_STREAM_RUNS_TABLE} (component_name, status, started_at);
CREATE INDEX IF NOT EXISTS idx_{CASCADE_STREAM_RESULTS_TABLE}_component
    ON {CASCADE_STREAM_RESULTS_TABLE} (component_name, created_at);
CREATE INDEX IF NOT EXISTS idx_{CASCADE_STREAM_RESULTS_TABLE}_run
    ON {CASCADE_STREAM_RESULTS_TABLE} (run_id);
"""


@dataclass(frozen=True)
class CascadeStreamRun:
    """A read-only view of one `cascade_stream_runs` row."""

    id: int
    component_name: str
    source: str
    source_detail: str
    started_at: str
    last_heartbeat_at: str
    finished_at: str | None
    status: str
    frames_seen: int
    frames_processed: int
    last_error: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "CascadeStreamRun":
        return cls(
            id=row["id"],
            component_name=row["component_name"],
            source=row["source"],
            source_detail=row["source_detail"],
            started_at=row["started_at"],
            last_heartbeat_at=row["last_heartbeat_at"],
            finished_at=row["finished_at"],
            status=row["status"],
            frames_seen=row["frames_seen"],
            frames_processed=row["frames_processed"],
            last_error=row["last_error"],
        )


@dataclass(frozen=True)
class CascadeStreamResult:
    """A read-only view of one `cascade_stream_results` row."""

    id: int
    run_id: int
    component_name: str
    created_at: str
    source: str
    frame_ref: str
    object_count: int
    objects: list[dict[str, Any]]
    thumbnail_path: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "CascadeStreamResult":
        return cls(
            id=row["id"],
            run_id=row["run_id"],
            component_name=row["component_name"],
            created_at=row["created_at"],
            source=row["source"],
            frame_ref=row["frame_ref"],
            object_count=row["object_count"],
            objects=json.loads(row["objects_json"] or "[]"),
            thumbnail_path=row["thumbnail_path"],
        )


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    for table in (CASCADE_STREAM_RUNS_TABLE, CASCADE_STREAM_RESULTS_TABLE):
        existing_columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for column, ddl in _MIGRATIONS:
            if column not in existing_columns:
                conn.execute(ddl)
    conn.executescript(_INDEX_DDL)
    # Must commit here, unconditionally — get_active_run()/list_runs()/
    # list_recent_results() reach this via config/database.py's
    # get_connection(), which does NOT auto-commit (unlike connect()'s
    # context manager, used by start_run()/record_result()/
    # delete_all_for_component()). Without this, schema creation triggered
    # by a read call silently rolls back when that connection closes —
    # the exact bug already found and fixed this session in
    # core/cascade/specialists/face/store.py, core/training_runs/store.py,
    # and core/inspections/store.py; applying the same fix here
    # preemptively rather than waiting to hit it again.
    conn.commit()


def start_run(component_name: str, *, source: str, source_detail: str = "") -> CascadeStreamRun:
    """Insert a new run row — called once when the Kafka consumer process
    starts, or once at the beginning of a video-file processing pass."""
    if source not in VALID_SOURCES:
        raise ValueError(f"Invalid source {source!r}; must be one of {VALID_SOURCES}")
    with connect(DB_PATH) as conn:
        _ensure_schema(conn)
        cursor = conn.execute(
            f"""INSERT INTO {CASCADE_STREAM_RUNS_TABLE} (component_name, source, source_detail)
                VALUES (?, ?, ?)""",
            (component_name, source, source_detail),
        )
        new_id = cursor.lastrowid
    record = get_run(new_id)
    assert record is not None
    return record


def heartbeat(run_id: int, *, frames_seen: int | None = None, frames_processed: int | None = None) -> None:
    """Update a running run's liveness timestamp and, optionally, its
    counters — called periodically (not per-frame) by the Kafka consumer
    loop, and once at the end of a bounded video pass."""
    assignments = ["last_heartbeat_at = datetime('now')"]
    params: list[Any] = []
    if frames_seen is not None:
        assignments.append("frames_seen = ?")
        params.append(frames_seen)
    if frames_processed is not None:
        assignments.append("frames_processed = ?")
        params.append(frames_processed)
    params.append(run_id)
    with connect(DB_PATH) as conn:
        _ensure_schema(conn)
        conn.execute(f"UPDATE {CASCADE_STREAM_RUNS_TABLE} SET {', '.join(assignments)} WHERE id = ?", params)


def finish_run(run_id: int, *, status: str, error: str | None = None) -> None:
    """Mark a run finished — 'stopped' (clean shutdown), 'crashed' (an
    uncaught exception, with `error` populated), or 'completed' (a bounded
    video pass reached the end of the file). Always sets `finished_at`, so
    a row's own timestamps alone say whether/how long it ran."""
    if status not in VALID_RUN_STATUSES or status == "running":
        raise ValueError(f"Invalid finish status {status!r}; must be one of {[s for s in VALID_RUN_STATUSES if s != 'running']}")
    with connect(DB_PATH) as conn:
        _ensure_schema(conn)
        conn.execute(
            f"""UPDATE {CASCADE_STREAM_RUNS_TABLE}
                SET status = ?, last_error = ?, finished_at = datetime('now'), last_heartbeat_at = datetime('now')
                WHERE id = ?""",
            (status, error, run_id),
        )


def get_run(run_id: int) -> CascadeStreamRun | None:
    conn = get_connection(DB_PATH)
    try:
        _ensure_schema(conn)
        row = conn.execute(f"SELECT * FROM {CASCADE_STREAM_RUNS_TABLE} WHERE id = ?", (run_id,)).fetchone()
    finally:
        conn.close()
    return CascadeStreamRun.from_row(row) if row else None


def get_active_run(component_name: str) -> CascadeStreamRun | None:
    """The most recent status='running' row for this component, if any.

    A "running" status here means only that nothing has explicitly stopped
    or finished it yet — it does NOT by itself mean the process is still
    alive (a crash can leave a row stuck at 'running' forever, since
    there's nothing left to write 'crashed' to it). Callers (the Cascade
    Stream page) must additionally check `last_heartbeat_at` against
    CASCADE_STREAM_HEARTBEAT_STALE_SECONDS to tell an actually-live run
    from a stale one.
    """
    conn = get_connection(DB_PATH)
    try:
        _ensure_schema(conn)
        row = conn.execute(
            f"""SELECT * FROM {CASCADE_STREAM_RUNS_TABLE}
                WHERE component_name = ? AND status = 'running'
                ORDER BY started_at DESC LIMIT 1""",
            (component_name,),
        ).fetchone()
    finally:
        conn.close()
    return CascadeStreamRun.from_row(row) if row else None


def list_runs(component_name: str, *, limit: int = 20) -> list[CascadeStreamRun]:
    conn = get_connection(DB_PATH)
    try:
        _ensure_schema(conn)
        rows = conn.execute(
            f"""SELECT * FROM {CASCADE_STREAM_RUNS_TABLE}
                WHERE component_name = ? ORDER BY started_at DESC LIMIT ?""",
            (component_name, limit),
        ).fetchall()
    finally:
        conn.close()
    return [CascadeStreamRun.from_row(r) for r in rows]


def _serialize_objects(objects: "list[DetectedObject]") -> str:
    """DetectedObject (and its nested SpecialistResult/PolicyExecutionResult/
    ReactionPolicy) -> JSON. `dataclasses.asdict()` recursively flattens the
    nested dataclasses to plain dicts, but doesn't make everything JSON-safe
    on its own — PolicyExecutionResult.saved_frame_path is a `Path | None`,
    the one field in this tree asdict() can't serialize — hence
    `default=str`. Tuples (box, executed_actions) serialize fine natively
    as JSON arrays."""
    return json.dumps([dataclasses.asdict(obj) for obj in objects], default=str)


def record_result(
    run_id: int,
    *,
    component_name: str,
    source: str,
    frame_ref: str,
    objects: "list[DetectedObject]",
    thumbnail_path: str | None = None,
) -> CascadeStreamResult:
    """Persist one processed frame's cascade output."""
    if source not in VALID_SOURCES:
        raise ValueError(f"Invalid source {source!r}; must be one of {VALID_SOURCES}")
    with connect(DB_PATH) as conn:
        _ensure_schema(conn)
        cursor = conn.execute(
            f"""INSERT INTO {CASCADE_STREAM_RESULTS_TABLE}
                (run_id, component_name, source, frame_ref, object_count, objects_json, thumbnail_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (run_id, component_name, source, frame_ref, len(objects), _serialize_objects(objects), thumbnail_path),
        )
        new_id = cursor.lastrowid
    record = get_result(new_id)
    assert record is not None
    return record


def get_result(result_id: int) -> CascadeStreamResult | None:
    conn = get_connection(DB_PATH)
    try:
        _ensure_schema(conn)
        row = conn.execute(f"SELECT * FROM {CASCADE_STREAM_RESULTS_TABLE} WHERE id = ?", (result_id,)).fetchone()
    finally:
        conn.close()
    return CascadeStreamResult.from_row(row) if row else None


def list_recent_results(component_name: str, *, limit: int = 50) -> list[CascadeStreamResult]:
    conn = get_connection(DB_PATH)
    try:
        _ensure_schema(conn)
        rows = conn.execute(
            f"""SELECT * FROM {CASCADE_STREAM_RESULTS_TABLE}
                WHERE component_name = ? ORDER BY created_at DESC, id DESC LIMIT ?""",
            (component_name, limit),
        ).fetchall()
    finally:
        conn.close()
    return [CascadeStreamResult.from_row(r) for r in rows]


def delete_results_for_component(component_name: str) -> int:
    """Manual cleanup for one component's accumulated stream thumbnails —
    the "Clean" button on the Cascade Stream page. Deletes every
    `cascade_stream_results` row for this component AND unlinks each
    row's thumbnail file on disk (best-effort: a file already missing is
    a safe no-op, never an error) — there is no automatic retention job
    for CASCADE_STREAM_FRAMES_DIR (see settings.py's own comment), so this
    is the only way that directory ever shrinks.

    Deliberately leaves `cascade_stream_runs` untouched — a run's own
    counters (frames_seen/frames_processed/status) are lightweight
    history, not disk-eating images, and stay meaningful as a record of
    "when did this stream run and how much did it see" even after its
    per-frame images are cleared. Use delete_all_for_component() instead
    (component_deletion.py's own use case) when the whole component,
    history included, is being permanently destroyed.

    Returns how many result rows were removed.
    """
    conn = get_connection(DB_PATH)
    try:
        _ensure_schema(conn)
        rows = conn.execute(
            f"SELECT thumbnail_path FROM {CASCADE_STREAM_RESULTS_TABLE} WHERE component_name = ?",
            (component_name,),
        ).fetchall()
    finally:
        conn.close()

    for row in rows:
        if row["thumbnail_path"]:
            Path(row["thumbnail_path"]).unlink(missing_ok=True)

    with connect(DB_PATH) as conn:
        _ensure_schema(conn)
        cursor = conn.execute(f"DELETE FROM {CASCADE_STREAM_RESULTS_TABLE} WHERE component_name = ?", (component_name,))
        return cursor.rowcount


def count_results_for_component(component_name: str) -> int:
    """Used by core/component_deletion.py's summarize_deletion_impact() —
    a plain count, not a fetch-and-len(), same reasoning
    SqliteMachineContextSource.count_readings() already applies."""
    conn = get_connection(DB_PATH)
    try:
        _ensure_schema(conn)
        row = conn.execute(
            f"SELECT COUNT(*) AS n FROM {CASCADE_STREAM_RESULTS_TABLE} WHERE component_name = ?",
            (component_name,),
        ).fetchone()
    finally:
        conn.close()
    return row["n"]


def delete_all_for_component(component_name: str) -> int:
    """Permanently remove every run and result row for this component in
    one call. Used exclusively by core/component_deletion.py's
    permanently_delete_component() — same one-shot cascade role
    core/inspections/store.py's and core/training_runs/store.py's own
    delete_all_for_component() play for their tables. Idempotent. Returns
    the total number of rows removed across both tables.
    """
    with connect(DB_PATH) as conn:
        _ensure_schema(conn)
        results_cursor = conn.execute(
            f"DELETE FROM {CASCADE_STREAM_RESULTS_TABLE} WHERE component_name = ?", (component_name,)
        )
        runs_cursor = conn.execute(
            f"DELETE FROM {CASCADE_STREAM_RUNS_TABLE} WHERE component_name = ?", (component_name,)
        )
        return results_cursor.rowcount + runs_cursor.rowcount
