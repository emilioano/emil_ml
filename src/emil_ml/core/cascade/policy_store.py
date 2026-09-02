"""Persistence for the reaction-policy table — SQLite table
`reaction_policies`, keyed by (component_name, specialist, identity_key),
following the same self-contained schema pattern as
core/training_runs/store.py and core/cascade/specialists/face/store.py.

Composite-keyed by component_name AND specialist, not just identity_key,
so: two different cascade components can react differently to the same
recognized person (see policy.py's module docstring for why that matters),
and "unknown" for the face specialist vs. "unknown" for a future car
specialist never collide within one component — see
core/cascade/base.py's BaseSpecialist.name and this module's own
upsert_policy().
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass

from emil_ml.config.database import connect, get_connection
from emil_ml.config.settings import DB_PATH
from emil_ml.core.cascade.policy import DEFAULT_PRIORITY, ReactionPolicy, VALID_ACTIONS, VALID_PRIORITIES

logger = logging.getLogger(__name__)

_TABLE = "reaction_policies"

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {_TABLE} (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    component_name TEXT NOT NULL,
    specialist     TEXT NOT NULL,
    identity_key   TEXT NOT NULL,
    label          TEXT NOT NULL,
    message        TEXT NOT NULL,
    actions        TEXT NOT NULL,
    priority       TEXT NOT NULL DEFAULT '{DEFAULT_PRIORITY}',
    UNIQUE(component_name, specialist, identity_key)
);
"""


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    _migrate_to_per_component(conn)
    # Must commit here, unconditionally — get_policy()/list_policies() reach
    # this via config/database.py's get_connection(), which does NOT
    # auto-commit (unlike connect()'s context manager, used by
    # upsert_policy()/delete_policy()). Without this, a migration triggered
    # by a read call silently rolls back when that connection closes — the
    # exact bug already found and fixed this session in
    # core/cascade/specialists/face/store.py, core/training_runs/store.py,
    # and core/inspections/store.py; applying the same fix here
    # preemptively rather than waiting to hit it again.
    conn.commit()


def _migrate_to_per_component(conn: sqlite3.Connection) -> None:
    """One-time, idempotent migration from the original globally-keyed
    (specialist, identity_key) shape to the current (component_name,
    specialist, identity_key) shape. A pre-migration row has no component
    association — it applied to every cascade component running that
    specialist — so the behavior-preserving migration is to duplicate it
    onto every CURRENTLY REGISTERED cascade-only component, so nothing an
    operator already configured silently stops reacting; a component
    created after this migration starts with no policies configured, same
    as any other new component would.

    Runs on every _ensure_schema() call but only ever does anything once
    per database: after the first run, `component_name` is  a real column
    with real values, so the "old shape" check below is false for good.
    """
    existing_columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({_TABLE})").fetchall()}
    if "component_name" in existing_columns:
        return  # already migrated (or a fresh DB that only ever knew the new shape)

    old_rows = conn.execute(f"SELECT * FROM {_TABLE}").fetchall()

    from emil_ml.config.registry import ComponentRegistry
    from emil_ml.core import registry_factory

    cascade_components = [
        c for c in ComponentRegistry().list_all() if registry_factory.is_cascade_only(c.model_type)
    ]

    conn.execute(f"ALTER TABLE {_TABLE} RENAME TO {_TABLE}_pre_component_scope")
    conn.executescript(_SCHEMA)
    migrated = 0
    for row in old_rows:
        for component in cascade_components:
            conn.execute(
                f"""INSERT OR IGNORE INTO {_TABLE}
                    (component_name, specialist, identity_key, label, message, actions, priority)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    component.name, row["specialist"], row["identity_key"],
                    row["label"], row["message"], row["actions"], row["priority"],
                ),
            )
            migrated += 1
    conn.execute(f"DROP TABLE {_TABLE}_pre_component_scope")
    logger.info(
        "migrated %d legacy reaction_policies row(s) onto %d existing cascade component(s) (%d row(s) written)",
        len(old_rows), len(cascade_components), migrated,
    )


def _row_to_policy(row: sqlite3.Row) -> ReactionPolicy:
    return ReactionPolicy(
        component_name=row["component_name"],
        specialist=row["specialist"],
        identity_key=row["identity_key"],
        label=row["label"],
        message=row["message"],
        actions=tuple(json.loads(row["actions"])),
        priority=row["priority"],
    )


def upsert_policy(
    component_name: str,
    specialist: str,
    identity_key: str,
    *,
    label: str,
    message: str,
    actions: list[str] | tuple[str, ...],
    priority: str = DEFAULT_PRIORITY,
) -> ReactionPolicy:
    """Create or replace the policy for (component_name, specialist, identity_key)."""
    unknown_actions = set(actions) - set(VALID_ACTIONS)
    if unknown_actions:
        raise ValueError(f"Invalid action(s) {sorted(unknown_actions)}; must be a subset of {VALID_ACTIONS}")
    if priority not in VALID_PRIORITIES:
        raise ValueError(f"Invalid priority {priority!r}; must be one of {VALID_PRIORITIES}")

    with connect(DB_PATH) as conn:
        _ensure_schema(conn)
        conn.execute(
            f"""INSERT INTO {_TABLE} (component_name, specialist, identity_key, label, message, actions, priority)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(component_name, specialist, identity_key) DO UPDATE SET
                    label = excluded.label, message = excluded.message,
                    actions = excluded.actions, priority = excluded.priority""",
            (component_name, specialist, identity_key, label, message, json.dumps(list(actions)), priority),
        )
    policy = get_policy(component_name, specialist, identity_key)
    assert policy is not None
    return policy


def get_policy(component_name: str, specialist: str, identity_key: str) -> ReactionPolicy | None:
    conn = get_connection(DB_PATH)
    try:
        _ensure_schema(conn)
        row = conn.execute(
            f"SELECT * FROM {_TABLE} WHERE component_name = ? AND specialist = ? AND identity_key = ?",
            (component_name, specialist, identity_key),
        ).fetchone()
    finally:
        conn.close()
    return _row_to_policy(row) if row else None


def list_policies(component_name: str, specialist: str | None = None) -> list[ReactionPolicy]:
    conn = get_connection(DB_PATH)
    try:
        _ensure_schema(conn)
        if specialist is None:
            rows = conn.execute(
                f"SELECT * FROM {_TABLE} WHERE component_name = ? ORDER BY specialist, identity_key",
                (component_name,),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT * FROM {_TABLE} WHERE component_name = ? AND specialist = ? ORDER BY identity_key",
                (component_name, specialist),
            ).fetchall()
    finally:
        conn.close()
    return [_row_to_policy(r) for r in rows]


def delete_policy(component_name: str, specialist: str, identity_key: str) -> None:
    with connect(DB_PATH) as conn:
        _ensure_schema(conn)
        conn.execute(
            f"DELETE FROM {_TABLE} WHERE component_name = ? AND specialist = ? AND identity_key = ?",
            (component_name, specialist, identity_key),
        )


def delete_all_for_component(component_name: str) -> int:
    """Permanently remove every policy row configured for this component —
    used by core/component_deletion.py's permanently_delete_component() so
    a deleted component's reaction policies don't linger as orphaned rows
    (and, since component names are reused after a permanent delete, don't
    silently apply to a future, unrelated component with the same name).
    Idempotent. Returns how many rows were removed.
    """
    with connect(DB_PATH) as conn:
        _ensure_schema(conn)
        cursor = conn.execute(f"DELETE FROM {_TABLE} WHERE component_name = ?", (component_name,))
        return cursor.rowcount
