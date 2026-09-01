"""Persistence for the `inspections` table — the queryable, durable
record of every inspection, independent of whether the analyzed image
file or its .report.md still exist on disk (see lifecycle.py for those).

The database is authoritative: a .report.md can always be regenerated
from a stored InspectionRecord; a stored record is never regenerated
from the .report.md file. Self-contained schema management (its own
_ensure_schema() call against the same emil.db), following the pattern
core/reporting/machine_context/source.py already established for a
domain-specific table that isn't a per-component setting.

`image_path`/`report_path` are stored relative to the owning component's
root (e.g. "analyzed/approved/20260808_..._017.png"), never absolute —
same portability reasoning as Component.model_path (see utils/paths.py's
ComponentPaths.resolve_model_path() and its docstring for the story
there): an absolute path baked in at inspection time ties the record to
whatever machine/filesystem produced it.

Prediction vs. verified truth are two separate, independent layers on the
same row, and that separation is deliberate, not incidental:
`verdict`/`score`/`defect_classes` are the model's own output and are
NEVER overwritten by anything in this module — they're the permanent
record of what the model actually said. `verified_status`/`verified_label`/
`verified_by`/`verified_at` are a human's later judgment about whether
that output was right, set only by verify() below. Nothing else in this
file — not create(), not acknowledge(), not mark_archived() — ever touches
the verified_* columns; every inspection starts and stays 'unverified'
until a human explicitly calls verify(). This is what makes
list_verified_for_training() a safe foundation for retraining: consuming
a model's own unverified guesses as if they were ground truth would let a
wrong prediction quietly become "confirmed" simply by going unchallenged,
a self-reinforcing feedback loop with no human ever actually in it.
Acknowledgement (including orchestrator.py's auto_acknowledge for
approved verdicts) is a workflow convenience — "this has been dealt
with" — never a truth claim, so it's kept off this axis entirely.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from emil_ml.config.database import connect, get_connection
from emil_ml.config.settings import DB_PATH, INSPECTIONS_TABLE

VALID_STATUSES = ("new", "acknowledged", "archived")
VALID_REPORT_STATUSES = ("none", "pending", "complete", "failed")

# A separate axis from `status`/report_status entirely — see verify()'s own
# docstring and the module docstring below for the full reasoning. "unverified"
# is the permanent default for every inspection until a human explicitly
# says otherwise; nothing else (not acknowledge(), not auto_acknowledge in
# orchestrator.py) is allowed to move a record off it.
VALID_VERIFIED_STATUSES = ("unverified", "verified_correct", "verified_incorrect")

# Automatically derived by verify() (see its own docstring) from prediction
# vs. verified_label — never set directly by a caller — so it's always
# consistent with the actual diff, not an independent claim that could
# drift from the label itself.
VALID_VERIFIED_ERROR_TYPES = ("false_positive", "false_negative", "wrong_class")

# A bookkeeping axis layered ON TOP of the verified_* columns, not part of
# them: whether a verified example has already been materialized into a
# component's training pool (see training/onboard.py's
# incorporate_verified_corrections()), so the same correction is
# never copied into that pool twice. Deliberately NOT written by verify()
# — see mark_incorporated() below, the only function allowed to touch it.
VALID_INCORPORATION_STATUSES = ("pending", "incorporated")

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {INSPECTIONS_TABLE} (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    component_name        TEXT NOT NULL,
    created_at            TEXT NOT NULL DEFAULT (datetime('now')),
    verdict               TEXT NOT NULL,
    score                 REAL NOT NULL,
    threshold             REAL,
    defect_classes        TEXT NOT NULL DEFAULT '[]',
    report_text           TEXT,
    report_sources        TEXT NOT NULL DEFAULT '[]',
    machine_context_used  TEXT NOT NULL DEFAULT '[]',
    report_model          TEXT,
    report_prompt         TEXT,
    report_thinking       TEXT,
    image_path            TEXT,
    report_path           TEXT,
    report_status         TEXT NOT NULL DEFAULT 'none',
    status                TEXT NOT NULL DEFAULT 'new',
    run_by                TEXT,
    acknowledged_by       TEXT,
    acknowledged_at       TEXT,
    archived_at           TEXT,
    verified_status       TEXT NOT NULL DEFAULT 'unverified',
    verified_label        TEXT,
    verified_by           TEXT,
    verified_at           TEXT,
    verified_error_type   TEXT,
    verified_incorporation_status TEXT NOT NULL DEFAULT 'pending'
);
CREATE INDEX IF NOT EXISTS idx_{INSPECTIONS_TABLE}_component_status
    ON {INSPECTIONS_TABLE} (component_name, status);
"""

# Indexed separately from _SCHEMA above, and created only after
# _MIGRATIONS run (see _ensure_schema()) rather than inline in _SCHEMA's
# executescript: on a database that already had `inspections` before
# verified_status existed, CREATE TABLE IF NOT EXISTS is a no-op, so an
# index on verified_status inside the same script would run against a
# table that doesn't have that column yet — the ALTER TABLE that adds it
# hasn't happened at that point in _ensure_schema() yet.
_VERIFIED_INDEX_DDL = (
    f"CREATE INDEX IF NOT EXISTS idx_{INSPECTIONS_TABLE}_component_verified "
    f"ON {INSPECTIONS_TABLE} (component_name, verified_status)"
)


@dataclass(frozen=True)
class InspectionRecord:
    """A read-only view of one `inspections` row."""

    id: int
    component_name: str
    created_at: str
    verdict: str
    score: float
    threshold: float | None
    defect_classes: list[str]
    report_text: str | None
    report_sources: list[dict[str, str]]
    machine_context_used: list[str]
    report_model: str | None
    report_prompt: str | None
    report_thinking: str | None
    image_path: str | None
    report_path: str | None
    report_status: str
    status: str
    run_by: str | None
    acknowledged_by: str | None
    acknowledged_at: str | None
    archived_at: str | None
    verified_status: str
    verified_label: dict[str, Any] | None
    verified_by: str | None
    verified_at: str | None
    verified_error_type: str | None
    verified_incorporation_status: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "InspectionRecord":
        return cls(
            id=row["id"],
            component_name=row["component_name"],
            created_at=row["created_at"],
            verdict=row["verdict"],
            score=row["score"],
            threshold=row["threshold"],
            defect_classes=json.loads(row["defect_classes"] or "[]"),
            report_text=row["report_text"],
            report_sources=json.loads(row["report_sources"] or "[]"),
            machine_context_used=json.loads(row["machine_context_used"] or "[]"),
            report_model=row["report_model"],
            report_prompt=row["report_prompt"],
            report_thinking=row["report_thinking"],
            image_path=row["image_path"],
            report_path=row["report_path"],
            report_status=row["report_status"],
            status=row["status"],
            run_by=row["run_by"],
            acknowledged_by=row["acknowledged_by"],
            acknowledged_at=row["acknowledged_at"],
            archived_at=row["archived_at"],
            verified_status=row["verified_status"],
            verified_label=json.loads(row["verified_label"]) if row["verified_label"] else None,
            verified_by=row["verified_by"],
            verified_at=row["verified_at"],
            verified_error_type=row["verified_error_type"],
            verified_incorporation_status=row["verified_incorporation_status"],
        )


# (column_name, DDL) pairs applied to databases created before that column
# existed — same pattern as config/database.py's _MIGRATIONS, kept
# separate here since this table manages its own schema (see module
# docstring).
_MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("report_prompt", f"ALTER TABLE {INSPECTIONS_TABLE} ADD COLUMN report_prompt TEXT"),
    ("report_thinking", f"ALTER TABLE {INSPECTIONS_TABLE} ADD COLUMN report_thinking TEXT"),
    ("run_by", f"ALTER TABLE {INSPECTIONS_TABLE} ADD COLUMN run_by TEXT"),
    (
        "verified_status",
        f"ALTER TABLE {INSPECTIONS_TABLE} ADD COLUMN verified_status TEXT NOT NULL DEFAULT 'unverified'",
    ),
    ("verified_label", f"ALTER TABLE {INSPECTIONS_TABLE} ADD COLUMN verified_label TEXT"),
    ("verified_by", f"ALTER TABLE {INSPECTIONS_TABLE} ADD COLUMN verified_by TEXT"),
    ("verified_at", f"ALTER TABLE {INSPECTIONS_TABLE} ADD COLUMN verified_at TEXT"),
    ("verified_error_type", f"ALTER TABLE {INSPECTIONS_TABLE} ADD COLUMN verified_error_type TEXT"),
    (
        "verified_incorporation_status",
        f"ALTER TABLE {INSPECTIONS_TABLE} ADD COLUMN verified_incorporation_status TEXT NOT NULL DEFAULT 'pending'",
    ),
)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    existing_columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({INSPECTIONS_TABLE})").fetchall()}
    for column, ddl in _MIGRATIONS:
        if column not in existing_columns:
            conn.execute(ddl)
    conn.execute(_VERIFIED_INDEX_DDL)
    # Must commit here, unconditionally — several callers (list_all(),
    # get(), ...) reach this via config/database.py's get_connection(),
    # which does NOT auto-commit (unlike connect()'s context manager).
    # Without this, a migration triggered by a read call silently rolls
    # back when that connection closes and gets fruitlessly re-attempted
    # on every subsequent read until a write finally commits it for real
    # — confirmed as a real, reproducible bug in the sibling
    # core/cascade/specialists/face/store.py migration; applied here too
    # for the same reason.
    conn.commit()


def create(
    component_name: str,
    *,
    verdict: str,
    score: float,
    threshold: float | None,
    defect_classes: list[str] | None = None,
    image_path: str | None = None,
    report_status: str = "none",
    run_by: str | None = None,
) -> InspectionRecord:
    """Insert a new inspection with status='new'. Called once detection
    has finished and the image has already been written to disk (see
    lifecycle.py) — image_path should point at a file that already
    exists by the time this is called, never the other way around.

    `run_by` records who/what ran this inspection — an operator name for a
    Streamlit upload, "watcher" for the folder watcher — distinct from
    `acknowledged_by`, which is always a human and set later, separately
    (see acknowledge() below). Nullable: older records and any caller that
    doesn't pass it just show as unattributed rather than failing.
    """
    if report_status not in VALID_REPORT_STATUSES:
        raise ValueError(f"Invalid report_status {report_status!r}; must be one of {VALID_REPORT_STATUSES}")
    with connect(DB_PATH) as conn:
        _ensure_schema(conn)
        cursor = conn.execute(
            f"""INSERT INTO {INSPECTIONS_TABLE}
                (component_name, verdict, score, threshold, defect_classes, image_path, report_status, run_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                component_name,
                verdict,
                score,
                threshold,
                json.dumps(defect_classes or []),
                image_path,
                report_status,
                run_by,
            ),
        )
        new_id = cursor.lastrowid
    record = get(new_id)
    assert record is not None
    return record


def get(inspection_id: int) -> InspectionRecord | None:
    conn = get_connection(DB_PATH)
    try:
        _ensure_schema(conn)
        row = conn.execute(f"SELECT * FROM {INSPECTIONS_TABLE} WHERE id = ?", (inspection_id,)).fetchone()
    finally:
        conn.close()
    return InspectionRecord.from_row(row) if row else None


def list_all(
    *,
    component_name: str | None = None,
    status: str | None = None,
    verdict: str | None = None,
    report_status: str | None = None,
    verified_status: str | None = None,
    limit: int | None = None,
) -> list[InspectionRecord]:
    """Always ordered newest-first (created_at DESC) — any other viewing
    order (e.g. the Inspection Station's failed-first sort) is a caller-
    side concern applied to this result, not a query option here, so
    filtering (this function) and sorting (the caller) stay independent.
    """
    conn = get_connection(DB_PATH)
    try:
        _ensure_schema(conn)
        query = f"SELECT * FROM {INSPECTIONS_TABLE}"
        conditions: list[str] = []
        params: list[Any] = []
        if component_name is not None:
            conditions.append("component_name = ?")
            params.append(component_name)
        if status is not None:
            conditions.append("status = ?")
            params.append(status)
        if verdict is not None:
            conditions.append("verdict = ?")
            params.append(verdict)
        if report_status is not None:
            conditions.append("report_status = ?")
            params.append(report_status)
        if verified_status is not None:
            conditions.append("verified_status = ?")
            params.append(verified_status)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        rows = conn.execute(query, params).fetchall()
    finally:
        conn.close()
    return [InspectionRecord.from_row(r) for r in rows]


def list_verified_for_training(component_name: str) -> list[InspectionRecord]:
    """Every inspection for `component_name` a human has actually verified
    — the one query path retraining is meant to consume, and the reason
    verified_status exists as its own axis at all (see module docstring).

    Deliberately narrow: both conditions below are real guards, not
    redundant belt-and-suspenders. verified_status must be
    'verified_correct'/'verified_incorrect' (never 'unverified', the
    default every inspection starts and stays at until verify() is called
    — including an auto-acknowledged approved one; acknowledgement is not
    verification, see orchestrator.py's auto_acknowledge and acknowledge()
    below). verified_label must be non-NULL because that's the actual
    ground truth a training pipeline would consume — verify() always sets
    both together (see its own docstring), so a row with a verified
    status but no label would mean the two got out of sync somehow; this
    query treats that as "not usable" rather than trusting the status
    flag alone.

    Deliberately NOT filtered on `status` (new/acknowledged/archived) —
    this is an intentional design invariant, not an oversight: a verified
    correction is exactly as valuable to a retrain whether it's still
    'new', already 'acknowledged', or already 'archived'. Archiving only
    moves a file to archive/ (see lifecycle.archive()); it never deletes
    anything, and image_path is kept correct through the move — so an
    archived verified inspection is fully as usable as any other. Do NOT
    add a "WHERE status != 'archived'" (or similar) condition here; that
    would silently start excluding verified corrections the moment they
    get archived, for no real reason. Retention deletion (the actual
    place old archived data can disappear) has its own, separate
    protection for exactly this data — see core/inspections/retention.py's
    cleanup_archived_inspections(), which refuses to delete a verified
    inspection until it's been marked 'incorporated'.
    """
    conn = get_connection(DB_PATH)
    try:
        _ensure_schema(conn)
        rows = conn.execute(
            f"""SELECT * FROM {INSPECTIONS_TABLE}
                WHERE component_name = ?
                  AND verified_status IN ('verified_correct', 'verified_incorrect')
                  AND verified_label IS NOT NULL
                ORDER BY created_at DESC""",
            (component_name,),
        ).fetchall()
    finally:
        conn.close()
    return [InspectionRecord.from_row(r) for r in rows]


def update_report(
    inspection_id: int,
    *,
    report_text: str,
    report_sources: list[dict[str, str]],
    machine_context_used: list[str],
    report_model: str | None,
    report_path: str | None,
    report_status: str = "complete",
    report_prompt: str | None = None,
    report_thinking: str | None = None,
) -> None:
    """Fills in the report fields once generation finishes — called from
    the background thread in orchestrator.py, potentially long after
    create() returned the fast verdict.

    report_prompt/report_thinking are debugging-only (the exact prompt
    sent to the LLM, and its raw chain-of-thought if it returned one) —
    see the "verbose" expander in app/pages/1_inspect.py and 3_inspection_station.py.
    """
    if report_status not in VALID_REPORT_STATUSES:
        raise ValueError(f"Invalid report_status {report_status!r}; must be one of {VALID_REPORT_STATUSES}")
    with connect(DB_PATH) as conn:
        _ensure_schema(conn)
        conn.execute(
            f"""UPDATE {INSPECTIONS_TABLE}
                SET report_text = ?, report_sources = ?, machine_context_used = ?,
                    report_model = ?, report_path = ?, report_status = ?,
                    report_prompt = ?, report_thinking = ?
                WHERE id = ?""",
            (
                report_text,
                json.dumps(report_sources),
                json.dumps(machine_context_used),
                report_model,
                report_path,
                report_status,
                report_prompt,
                report_thinking,
                inspection_id,
            ),
        )


def mark_report_failed(inspection_id: int, error: str) -> None:
    """Degrade honestly instead of leaving report_status stuck on 'pending'
    forever — same "never crash, say so" principle llm.py's Ollama error
    handling already applies, just persisted this time."""
    with connect(DB_PATH) as conn:
        _ensure_schema(conn)
        conn.execute(
            f"UPDATE {INSPECTIONS_TABLE} SET report_text = ?, report_status = 'failed' WHERE id = ?",
            (f"Report generation failed: {error}", inspection_id),
        )


def acknowledge(inspection_id: int, *, by: str) -> None:
    """new -> acknowledged. Not a file move — see lifecycle.archive() for
    that, a deliberately separate step so there's an undo window between
    an operator's acknowledgement and the files actually being relocated.
    """
    with connect(DB_PATH) as conn:
        _ensure_schema(conn)
        conn.execute(
            f"""UPDATE {INSPECTIONS_TABLE}
                SET status = 'acknowledged', acknowledged_by = ?, acknowledged_at = datetime('now')
                WHERE id = ? AND status = 'new'""",
            (by, inspection_id),
        )


def bulk_acknowledge(inspection_ids: list[int], *, by: str) -> int:
    """Acknowledge every given inspection currently 'new', in a single
    transaction — the bulk-scale counterpart to acknowledge() above, same
    semantics (only flips status; files untouched) and the same
    only-if-still-'new' WHERE-clause guard, just batched instead of one
    connection per record. Exists specifically for the "thousands of
    approved records" case the Inspection Station's bulk actions handle —
    acknowledge() itself is unchanged and still the right call for a
    single record.

    Ids not currently 'new' (already acknowledged/archived, or simply not
    found) are silently skipped, same as acknowledge()'s own WHERE clause
    already does one at a time — never an error, since a bulk selection
    naturally includes records in different states. Returns how many rows
    were actually flipped, for the caller to report back.
    """
    if not inspection_ids:
        return 0
    with connect(DB_PATH) as conn:
        _ensure_schema(conn)
        placeholders = ", ".join("?" * len(inspection_ids))
        cursor = conn.execute(
            f"""UPDATE {INSPECTIONS_TABLE}
                SET status = 'acknowledged', acknowledged_by = ?, acknowledged_at = datetime('now')
                WHERE status = 'new' AND id IN ({placeholders})""",
            (by, *inspection_ids),
        )
        return cursor.rowcount


def revert_to_new(inspection_id: int) -> None:
    """acknowledged -> new: undoes an acknowledgement made in error (wrong
    button, wrong record) or reopened for a second look. Only a DB status
    flip — files are untouched either way at 'acknowledged', same as
    acknowledge() itself (see its own docstring) — so there's nothing to
    move back.

    Deliberately does NOT accept an 'archived' record: archiving is a
    physical file move (see lifecycle.archive()), not just a status flag,
    so there's no DB-only way to undo it — the WHERE clause below silently
    no-ops for any status other than 'acknowledged', same defensive
    pattern acknowledge() already uses for 'new'.

    Clears acknowledged_by/acknowledged_at rather than leaving them stale:
    if this inspection gets acknowledged again later, that's a fresh
    acknowledgement and deserves its own attribution/timestamp, not the
    reverted one still showing.
    """
    with connect(DB_PATH) as conn:
        _ensure_schema(conn)
        conn.execute(
            f"""UPDATE {INSPECTIONS_TABLE}
                SET status = 'new', acknowledged_by = NULL, acknowledged_at = NULL
                WHERE id = ? AND status = 'acknowledged'""",
            (inspection_id,),
        )


def _classify_error(prediction_verdict: str, prediction_defect_classes: list[str], label: dict[str, Any]) -> str | None:
    """What kind of mistake `label` (a human's corrected truth) reveals
    about the model's own prediction — or None if the label actually
    matches the prediction (not a real error). Purely derived, never an
    independent input — see verify()'s own docstring for why that matters.
    """
    label_verdict = label.get("verdict")
    label_classes = set(label.get("defect_classes") or [])
    if prediction_verdict == "approved" and label_verdict == "failed":
        return "false_negative"
    if prediction_verdict == "failed" and label_verdict == "approved":
        return "false_positive"
    if prediction_verdict == "failed" and label_verdict == "failed" and set(prediction_defect_classes) != label_classes:
        return "wrong_class"
    return None


def verify(inspection_id: int, *, status: str, label: dict[str, Any], by: str) -> None:
    """Record a human's judgment about whether the model's own prediction
    (verdict/score/defect_classes, untouched by this or any other function
    here — see module docstring) was actually right, and what the correct
    label is either way. This is the ONLY function in this module allowed
    to touch verified_status/verified_label/verified_by/verified_at/
    verified_error_type — acknowledge()/revert_to_new()/mark_archived() are
    workflow state, not truth claims, and never go near these columns.
    (verified_incorporation_status is a separate, later concern — see
    mark_incorporated() below, which this function never touches either.)

    `status` must be 'verified_correct' or 'verified_incorrect' — never
    'unverified' through here; that's the untouched default every
    inspection starts at, not something to explicitly (re-)set.

    `label` is required even when `status` is 'verified_correct': there's
    always a definite ground truth once a human has looked, and requiring
    it explicitly (rather than defaulting to "same as the prediction" when
    confirming) keeps list_verified_for_training()'s contract simple and
    absolute — every row it returns has a real, explicit label, full stop,
    never an implicit "go look at the prediction fields instead". A
    typical shape: {"verdict": "failed", "defect_classes": ["missing_bristles"],
    "boxes": [{"class": "missing_bristles", "box": [x1, y1, x2, y2]}]} for
    a failed unit, or {"verdict": "approved", "defect_classes": [], "boxes": []}
    for a confirmed-good one (including a corrected false positive) — the
    exact shape is a training-consumer concern, not something this layer
    validates beyond "it's a dict".

    verified_error_type (false_positive/false_negative/wrong_class) is
    computed automatically from the existing prediction vs. `label`, never
    taken as a separate parameter — a caller asserting an error type that
    doesn't match what the label actually says would be exactly the kind
    of drift this whole layer exists to prevent. For 'verified_incorrect',
    the comparison must reveal a real difference (raises ValueError if
    `label` actually matches the prediction — that's a contradiction, not
    a correction). For 'verified_correct' it's always None.
    """
    if status not in ("verified_correct", "verified_incorrect"):
        raise ValueError(
            f"Invalid verified status {status!r}; must be 'verified_correct' or 'verified_incorrect' "
            "('unverified' is the default and is never set explicitly)"
        )
    if not label:
        raise ValueError("A verified label is required — see verify()'s own docstring for why.")

    record = get(inspection_id)
    if record is None:
        raise KeyError(f"No inspection with id {inspection_id!r}")

    error_type = _classify_error(record.verdict, record.defect_classes, label)
    if status == "verified_incorrect" and error_type is None:
        raise ValueError(
            f"label {label!r} matches inspection {inspection_id}'s existing prediction "
            f"(verdict={record.verdict!r}, defect_classes={record.defect_classes!r}) — "
            "'verified_incorrect' requires a label that actually differs from the prediction; "
            "use status='verified_correct' to confirm a prediction as-is instead."
        )
    if status == "verified_correct":
        error_type = None

    with connect(DB_PATH) as conn:
        _ensure_schema(conn)
        conn.execute(
            f"""UPDATE {INSPECTIONS_TABLE}
                SET verified_status = ?, verified_label = ?, verified_by = ?, verified_at = datetime('now'),
                    verified_error_type = ?
                WHERE id = ?""",
            (status, json.dumps(label), by, error_type, inspection_id),
        )


def unverify(inspection_id: int) -> None:
    """Undo a verify() call made by mistake — a misclick on "Verify as
    correct" or the wrong "Flag as incorrect" option — by resetting
    verified_status back to its 'unverified' default and clearing
    verified_label/verified_by/verified_at/verified_error_type, so the
    record looks exactly like it did before anyone verified it and can be
    re-flagged/re-verified cleanly.

    Also resets verified_incorporation_status back to 'pending': that
    column only means anything relative to an actual verification, so an
    unverified record has nothing to be 'incorporated' about. One honest
    caveat this can't undo: if incorporate_verified_corrections()
    already materialized the old (mistaken) label into the component's
    training pool as real image/label files before this was caught,
    those files are still sitting there — this only fixes the DB record,
    not any pool files a training run already copied. In practice this
    is rarely a problem since a misclick is usually caught and undone
    long before the next retrain consumes it.
    """
    with connect(DB_PATH) as conn:
        _ensure_schema(conn)
        conn.execute(
            f"""UPDATE {INSPECTIONS_TABLE}
                SET verified_status = 'unverified', verified_label = NULL, verified_by = NULL,
                    verified_at = NULL, verified_error_type = NULL, verified_incorporation_status = 'pending'
                WHERE id = ?""",
            (inspection_id,),
        )


def mark_incorporated(inspection_ids: list[int]) -> None:
    """Flip verified_incorporation_status 'pending' -> 'incorporated' for
    the given inspections, once their verified_label has actually been
    materialized into a component's training pool (see training/onboard.py's
    incorporate_verified_corrections()) — so a later consumption pass
    never copies the same correction into that pool a second time.

    Deliberately separate from verify(): this is bookkeeping about
    whether a verification has been USED for training, not a claim about
    whether it's TRUE — the two are different axes on purpose (see
    module docstring), so this function only ever touches
    verified_incorporation_status, never verified_status/verified_label/
    verified_by/verified_at/verified_error_type.

    No-op for an empty list, so a caller doesn't need to special-case
    "nothing was actually consumed this run".
    """
    if not inspection_ids:
        return
    with connect(DB_PATH) as conn:
        _ensure_schema(conn)
        placeholders = ", ".join("?" * len(inspection_ids))
        conn.execute(
            f"UPDATE {INSPECTIONS_TABLE} SET verified_incorporation_status = 'incorporated' "
            f"WHERE id IN ({placeholders})",
            inspection_ids,
        )


def mark_archived(inspection_id: int, *, image_path: str | None, report_path: str | None) -> None:
    """Called by lifecycle.archive() in the same operation as the actual
    file move, so this record is never left pointing at a pre-move path.
    """
    with connect(DB_PATH) as conn:
        _ensure_schema(conn)
        conn.execute(
            f"""UPDATE {INSPECTIONS_TABLE}
                SET status = 'archived', archived_at = datetime('now'), image_path = ?, report_path = ?
                WHERE id = ?""",
            (image_path, report_path, inspection_id),
        )


def delete(inspection_id: int) -> None:
    """Permanently remove one inspection row. One of only two functions
    in this entire module that actually delete a row (delete_all_for_component()
    below is the other) — every other function here (acknowledge,
    archive, revert_to_new, unverify, ...) only ever moves a record
    between states, never removes it. This one exists for exactly one
    caller: core/inspections/retention.py's cleanup_archived_inspections(),
    the deliberate, explicit "archived -> deleted" step (the third and
    final lifecycle stage) — not something any workflow action should
    ever call directly. The caller is responsible for having already
    removed (or intentionally left) any files this record pointed to;
    this only touches the database row.
    """
    with connect(DB_PATH) as conn:
        _ensure_schema(conn)
        conn.execute(f"DELETE FROM {INSPECTIONS_TABLE} WHERE id = ?", (inspection_id,))


def delete_all_for_component(component_name: str) -> int:
    """Permanently remove EVERY inspection row for this component in one
    call — including every verified_* column, since verified_label etc.
    are columns on these same rows, not a separate table. Used exclusively
    by core/component_deletion.py's permanently_delete_component() when a
    whole component is being permanently destroyed (only ever reachable
    after that component was already soft-deleted — see
    ComponentRegistry.soft_delete()), never as a normal workflow action.
    Idempotent: deleting 0 remaining rows (e.g. a resumed, already-
    completed deletion) is a safe no-op. Returns how many rows were
    removed.
    """
    with connect(DB_PATH) as conn:
        _ensure_schema(conn)
        cursor = conn.execute(f"DELETE FROM {INSPECTIONS_TABLE} WHERE component_name = ?", (component_name,))
        return cursor.rowcount
