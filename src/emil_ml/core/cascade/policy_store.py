"""Persistence for the reaction-policy table — SQLite table
`reaction_policies`, keyed by (specialist, identity_key), following the
same self-contained schema pattern as core/training_runs/store.py and
core/cascade/specialists/face/store.py.

Composite-keyed by specialist, not just identity_key, so "unknown" for
the face specialist and "unknown" for a future car specialist never
collide — see core/cascade/base.py's BaseSpecialist.name and this
module's own upsert_policy().
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from emil_ml.config.database import connect, get_connection
from emil_ml.config.settings import DB_PATH
from emil_ml.core.cascade.policy import DEFAULT_PRIORITY, ReactionPolicy, VALID_ACTIONS, VALID_PRIORITIES

_TABLE = "reaction_policies"

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {_TABLE} (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    specialist   TEXT NOT NULL,
    identity_key TEXT NOT NULL,
    label        TEXT NOT NULL,
    message      TEXT NOT NULL,
    actions      TEXT NOT NULL,
    priority     TEXT NOT NULL DEFAULT '{DEFAULT_PRIORITY}',
    UNIQUE(specialist, identity_key)
);
"""


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)


def _row_to_policy(row: sqlite3.Row) -> ReactionPolicy:
    return ReactionPolicy(
        specialist=row["specialist"],
        identity_key=row["identity_key"],
        label=row["label"],
        message=row["message"],
        actions=tuple(json.loads(row["actions"])),
        priority=row["priority"],
    )


def upsert_policy(
    specialist: str,
    identity_key: str,
    *,
    label: str,
    message: str,
    actions: list[str] | tuple[str, ...],
    priority: str = DEFAULT_PRIORITY,
) -> ReactionPolicy:
    """Create or replace the policy for (specialist, identity_key)."""
    unknown_actions = set(actions) - set(VALID_ACTIONS)
    if unknown_actions:
        raise ValueError(f"Invalid action(s) {sorted(unknown_actions)}; must be a subset of {VALID_ACTIONS}")
    if priority not in VALID_PRIORITIES:
        raise ValueError(f"Invalid priority {priority!r}; must be one of {VALID_PRIORITIES}")

    with connect(DB_PATH) as conn:
        _ensure_schema(conn)
        conn.execute(
            f"""INSERT INTO {_TABLE} (specialist, identity_key, label, message, actions, priority)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(specialist, identity_key) DO UPDATE SET
                    label = excluded.label, message = excluded.message,
                    actions = excluded.actions, priority = excluded.priority""",
            (specialist, identity_key, label, message, json.dumps(list(actions)), priority),
        )
    policy = get_policy(specialist, identity_key)
    assert policy is not None
    return policy


def get_policy(specialist: str, identity_key: str) -> ReactionPolicy | None:
    conn = get_connection(DB_PATH)
    try:
        _ensure_schema(conn)
        row = conn.execute(
            f"SELECT * FROM {_TABLE} WHERE specialist = ? AND identity_key = ?", (specialist, identity_key)
        ).fetchone()
    finally:
        conn.close()
    return _row_to_policy(row) if row else None


def list_policies(specialist: str | None = None) -> list[ReactionPolicy]:
    conn = get_connection(DB_PATH)
    try:
        _ensure_schema(conn)
        if specialist is None:
            rows = conn.execute(f"SELECT * FROM {_TABLE} ORDER BY specialist, identity_key").fetchall()
        else:
            rows = conn.execute(
                f"SELECT * FROM {_TABLE} WHERE specialist = ? ORDER BY identity_key", (specialist,)
            ).fetchall()
    finally:
        conn.close()
    return [_row_to_policy(r) for r in rows]


def delete_policy(specialist: str, identity_key: str) -> None:
    with connect(DB_PATH) as conn:
        _ensure_schema(conn)
        conn.execute(f"DELETE FROM {_TABLE} WHERE specialist = ? AND identity_key = ?", (specialist, identity_key))
