"""Persistence for the face-recognition specialist's known-individuals
database — two SQLite tables, following the same self-contained schema
pattern as core/training_runs/store.py (its own `_SCHEMA` +
`_ensure_schema()`, no dependency on config/database.py's central
`components` schema):

- `known_individuals`: one row per consenting person — name, consent,
  identity_key.
- `known_face_embeddings`: MANY rows per person, one per registered
  photo. A single reference photo gives one embedding taken under one
  set of conditions (angle, lighting, expression), which makes matching
  brittle against real-world variation and gives nothing to calibrate a
  threshold against. Multiple photos per person — ideally taken under
  varying conditions — give both: a richer match target (see
  find_best_match()'s min-distance-to-any-of-their-embeddings strategy)
  and real observed intra-person/inter-person distance distributions to
  calibrate DEFAULT_FACE_MATCH_DISTANCE_THRESHOLD against instead of
  guessing (see calibration_stats() below) — the same principle
  core/evaluation's anomaly-score threshold calibration already
  established for the unsupervised image methods (autoencoder/PatchCore/
  Isolation Forest): a threshold should sit between two observed
  distributions, not be picked blind.

PRIVACY BY CONSTRUCTION — these tables are the ONLY source of identity
for the face specialist (see predictor.py's match logic): a face
embedding that doesn't come within DEFAULT_FACE_MATCH_DISTANCE_THRESHOLD
of ANY registered embedding is always "unknown". There is no path by
which this system can name someone who isn't registered — identification
is opt-in per person, not inferred from the frame. `add_known_individual()`'s
`consented` parameter is required with no default (rather than
`consented: bool = True`) specifically so a caller cannot register a
person by omission/oversight without consciously passing
`consented=True`; passing anything else raises. Consent is recorded once,
on the PERSON, and covers every one of their photos — adding more photos
via add_face_embedding() never re-asks for it and never needs to, since
withdrawing consent removes the person (and therefore every embedding of
theirs) in one action; see delete_known_individual().

Embeddings are 512-dim float vectors (facenet-pytorch's InceptionResnetV1,
vggface2 weights) — biometric data. Stored locally in this project's own
SQLite file only (no external service, no cloud sync), scoped to these two
tables; nothing outside core/cascade/specialists/face/ reads them
directly. Deliberately NOT storing the source photos or face crops
themselves alongside the embeddings — only the derived vector — to keep
each person's stored biometric footprint to the minimum this system
actually needs to function, even though that means the "remove one photo"
UI can only show a photo's registration timestamp, not a thumbnail of it.
`delete_known_individual()`/`delete_face_embedding()` are both full,
permanent row removal — the correct way to revoke consent (partially or
entirely), not a soft-delete/status flag (unlike ComponentRegistry, there
is no "trash" state here: once consent is withdrawn, the embedding(s)
should not linger).
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass

from emil_ml.config.database import connect, get_connection
from emil_ml.config.settings import DB_PATH

_INDIVIDUALS_TABLE = "known_individuals"
_EMBEDDINGS_TABLE = "known_face_embeddings"
_LEGACY_TABLE = "known_faces"  # pre-multi-photo shape: one row = one person = one embedding

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {_INDIVIDUALS_TABLE} (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    identity_key TEXT NOT NULL UNIQUE,
    name         TEXT NOT NULL,
    consented    INTEGER NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    notes        TEXT
);
CREATE TABLE IF NOT EXISTS {_EMBEDDINGS_TABLE} (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    identity_key TEXT NOT NULL,
    embedding    TEXT NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_{_EMBEDDINGS_TABLE}_identity
    ON {_EMBEDDINGS_TABLE} (identity_key);
"""


@dataclass(frozen=True)
class FaceEmbeddingRecord:
    """One registered photo's embedding — a read-only view of one
    `known_face_embeddings` row."""

    id: int
    identity_key: str
    embedding: list[float]
    created_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "FaceEmbeddingRecord":
        return cls(
            id=row["id"], identity_key=row["identity_key"],
            embedding=json.loads(row["embedding"]), created_at=row["created_at"],
        )


@dataclass(frozen=True)
class KnownIndividual:
    """A read-only view of one `known_individuals` row, plus how many
    photos/embeddings they currently have registered — the number the
    Onboard page's registration UI shows directly, without a second query."""

    id: int
    identity_key: str
    name: str
    consented: bool
    created_at: str
    notes: str | None
    embedding_count: int

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "KnownIndividual":
        return cls(
            id=row["id"],
            identity_key=row["identity_key"],
            name=row["name"],
            consented=bool(row["consented"]),
            created_at=row["created_at"],
            notes=row["notes"],
            embedding_count=row["embedding_count"],
        )


# Every SELECT that needs to produce a KnownIndividual joins in its own
# embedding count this same way, so from_row() never has to special-case
# "count wasn't selected" — one query shape, used everywhere a full
# KnownIndividual is read back.
_SELECT_INDIVIDUALS = f"""
    SELECT i.*, (SELECT COUNT(*) FROM {_EMBEDDINGS_TABLE} e WHERE e.identity_key = i.identity_key) AS embedding_count
    FROM {_INDIVIDUALS_TABLE} i
"""


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    _migrate_legacy_single_embedding_table(conn)
    # Schema creation/migration must be durable the instant it runs,
    # regardless of which caller triggered it — including a read-only one
    # (get_by_identity_key(), list_known_individuals(), find_best_match(),
    # ... all call this first, via config/database.py's get_connection(),
    # which does NOT auto-commit, unlike connect()'s context manager).
    # Confirmed the hard way: without this, migrating the legacy
    # known_faces table via a read path executed the DROP TABLE and row
    # copies, then silently rolled them all back when that connection
    # closed uncommitted — the very next call saw known_faces again and
    # re-ran the "migration" from scratch. Committing here, unconditionally,
    # is what config/database.py's own init_db() already does for exactly
    # this reason.
    conn.commit()


def _migrate_legacy_single_embedding_table(conn: sqlite3.Connection) -> None:
    """One-time, idempotent migration from the original one-row-per-person
    `known_faces` table (person + their single embedding in one row) to
    the current two-table shape. Runs on every _ensure_schema() call but
    only ever does anything once per database: it drops `known_faces`
    itself as its last step, inside the same connection, so a database
    that's already been migrated has nothing left to detect and this
    becomes a fast no-op (a single sqlite_master lookup) for good.
    """
    exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (_LEGACY_TABLE,)
    ).fetchone()
    if not exists:
        return

    import logging

    logger = logging.getLogger(__name__)
    legacy_rows = conn.execute(f"SELECT * FROM {_LEGACY_TABLE}").fetchall()
    for row in legacy_rows:
        conn.execute(
            f"""INSERT OR IGNORE INTO {_INDIVIDUALS_TABLE}
                (identity_key, name, consented, created_at, notes) VALUES (?, ?, ?, ?, ?)""",
            (row["identity_key"], row["name"], row["consented"], row["created_at"], row["notes"]),
        )
        conn.execute(
            f"INSERT INTO {_EMBEDDINGS_TABLE} (identity_key, embedding, created_at) VALUES (?, ?, ?)",
            (row["identity_key"], row["embedding"], row["created_at"]),
        )
    conn.execute(f"DROP TABLE {_LEGACY_TABLE}")
    logger.info(
        "migrated %d legacy known_faces row(s) into %s/%s (one embedding each, preserved as-is)",
        len(legacy_rows), _INDIVIDUALS_TABLE, _EMBEDDINGS_TABLE,
    )


def _slugify(name: str) -> str:
    from emil_ml.utils.paths import slugify

    return slugify(name)


def add_known_individual(
    name: str, embedding: list[float], *, consented: bool, notes: str | None = None
) -> KnownIndividual:
    """Register a new consenting individual with their FIRST photo's
    embedding — no model retraining, no image storage; the person row
    plus one embedding row. Calling this again for an already-registered
    name (same identity_key) refreshes their name/notes and ADDS another
    embedding, same as calling add_face_embedding() directly — so
    "register Alice with 3 photos" is just three calls in a row, and
    "register Alice now, add 2 more photos later" is this function once
    followed by add_face_embedding() twice.

    `consented` has no default on purpose (see module docstring): a
    caller must explicitly pass `consented=True`. Passing `False` (or
    anything else falsy) raises rather than silently no-op-ing, so a
    bug that reaches this function with the wrong flag fails loudly
    instead of quietly skipping the consent gate.
    """
    if consented is not True:
        raise ValueError(
            "consented must be True to add a known individual — this system only ever "
            "identifies people who have explicitly consented (see store.py's module docstring)"
        )
    if not name.strip():
        raise ValueError("name cannot be empty")
    if not embedding:
        raise ValueError("embedding cannot be empty")

    identity_key = _slugify(name)
    with connect(DB_PATH) as conn:
        _ensure_schema(conn)
        conn.execute(
            f"""INSERT INTO {_INDIVIDUALS_TABLE} (identity_key, name, consented, notes)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(identity_key) DO UPDATE SET
                    name = excluded.name, consented = excluded.consented, notes = excluded.notes""",
            (identity_key, name.strip(), int(consented), notes),
        )
        conn.execute(
            f"INSERT INTO {_EMBEDDINGS_TABLE} (identity_key, embedding) VALUES (?, ?)",
            (identity_key, json.dumps(embedding)),
        )
    individual = get_by_identity_key(identity_key)
    assert individual is not None
    return individual


def add_face_embedding(identity_key: str, embedding: list[float]) -> FaceEmbeddingRecord:
    """Add one more photo's embedding to an ALREADY-registered individual
    — the mechanism behind "add more photos later". Consent was already
    given when the person was first registered (see module docstring: it
    covers the person, not one photo) so this never asks for it again.
    Raises KeyError if `identity_key` isn't a registered individual —
    use add_known_individual() to register the person with their first
    photo first.
    """
    if not embedding:
        raise ValueError("embedding cannot be empty")
    if get_by_identity_key(identity_key) is None:
        raise KeyError(f"No known individual {identity_key!r} — register them first via add_known_individual()")

    with connect(DB_PATH) as conn:
        _ensure_schema(conn)
        cursor = conn.execute(
            f"INSERT INTO {_EMBEDDINGS_TABLE} (identity_key, embedding) VALUES (?, ?)",
            (identity_key, json.dumps(embedding)),
        )
        new_id = cursor.lastrowid
    conn = get_connection(DB_PATH)
    try:
        row = conn.execute(f"SELECT * FROM {_EMBEDDINGS_TABLE} WHERE id = ?", (new_id,)).fetchone()
    finally:
        conn.close()
    assert row is not None
    return FaceEmbeddingRecord.from_row(row)


def get_by_identity_key(identity_key: str) -> KnownIndividual | None:
    conn = get_connection(DB_PATH)
    try:
        _ensure_schema(conn)
        row = conn.execute(f"{_SELECT_INDIVIDUALS} WHERE i.identity_key = ?", (identity_key,)).fetchone()
    finally:
        conn.close()
    return KnownIndividual.from_row(row) if row else None


def list_known_individuals() -> list[KnownIndividual]:
    conn = get_connection(DB_PATH)
    try:
        _ensure_schema(conn)
        rows = conn.execute(f"{_SELECT_INDIVIDUALS} ORDER BY i.name").fetchall()
    finally:
        conn.close()
    return [KnownIndividual.from_row(r) for r in rows]


def list_embeddings_for(identity_key: str) -> list[FaceEmbeddingRecord]:
    """Every registered photo's embedding for one individual, oldest
    first — what the Onboard page's per-photo removal UI lists."""
    conn = get_connection(DB_PATH)
    try:
        _ensure_schema(conn)
        rows = conn.execute(
            f"SELECT * FROM {_EMBEDDINGS_TABLE} WHERE identity_key = ? ORDER BY created_at", (identity_key,)
        ).fetchall()
    finally:
        conn.close()
    return [FaceEmbeddingRecord.from_row(r) for r in rows]


def delete_face_embedding(embedding_id: int) -> None:
    """Remove ONE registered photo's embedding — the person and their
    other embeddings are untouched. A person can end up with zero
    embeddings this way (still registered/consented, just with nothing
    to match against yet) — a valid, if unusual, state; add_face_embedding()
    is how they get a reference photo again."""
    with connect(DB_PATH) as conn:
        _ensure_schema(conn)
        conn.execute(f"DELETE FROM {_EMBEDDINGS_TABLE} WHERE id = ?", (embedding_id,))


def delete_known_individual(identity_key: str) -> None:
    """Permanently remove a person AND every embedding of theirs — the
    correct way to revoke consent entirely (see module docstring: no
    soft-delete here). Removing just one of their photos instead is
    delete_face_embedding()."""
    with connect(DB_PATH) as conn:
        _ensure_schema(conn)
        conn.execute(f"DELETE FROM {_EMBEDDINGS_TABLE} WHERE identity_key = ?", (identity_key,))
        conn.execute(f"DELETE FROM {_INDIVIDUALS_TABLE} WHERE identity_key = ?", (identity_key,))


def _l2_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def find_best_match(embedding: list[float], *, threshold: float) -> tuple[KnownIndividual, float] | None:
    """Nearest-neighbor lookup by L2 distance against EVERY registered
    embedding of EVERY individual — min-distance-to-any-of-their-photos,
    not distance-to-a-centroid. Chosen deliberately: a person's photos
    taken under different conditions (angle, lighting, expression) can
    sit quite far apart in embedding space even though they're all
    genuinely that person, and averaging them into one centroid can land
    in a region none of the individual photos are actually close to,
    losing exactly the coverage multiple photos were meant to add.
    Min-distance-to-any preserves that coverage: a new photo only needs
    to be close to ONE of the person's registered views, not to their
    average. A linear scan over every embedding is appropriate at this
    scale (a course-demo-sized roster of consenting individuals with a
    handful of photos each, not a large-scale biometric system that
    would need approximate nearest-neighbor infrastructure).

    Returns (best-matching KnownIndividual, distance) only if that
    distance is below `threshold`; otherwise None — a caller
    (predictor.py) treats None as "unknown", never as an error. An
    individual with zero registered embeddings (see delete_face_embedding())
    simply never wins a match — nothing special-cased for it.
    """
    conn = get_connection(DB_PATH)
    try:
        _ensure_schema(conn)
        rows = conn.execute(f"SELECT * FROM {_EMBEDDINGS_TABLE}").fetchall()
    finally:
        conn.close()
    if not rows:
        return None

    embeddings = [FaceEmbeddingRecord.from_row(r) for r in rows]
    scored = [(e.identity_key, _l2_distance(embedding, e.embedding)) for e in embeddings]
    best_identity_key, best_distance = min(scored, key=lambda pair: pair[1])
    if best_distance >= threshold:
        return None
    best_individual = get_by_identity_key(best_identity_key)
    assert best_individual is not None
    return best_individual, best_distance


@dataclass(frozen=True)
class CalibrationStats:
    """Observed embedding-distance distributions across every currently
    registered individual — see calibration_stats()."""

    intra_person_distances: list[float]  # distances between the SAME person's own embeddings
    inter_person_distances: list[float]  # nearest-embedding distance between DIFFERENT people
    suggested_threshold: float | None  # None if there isn't enough data to suggest one (see calibration_stats())
    separable: bool  # False if the two distributions actually overlap (max intra >= min inter) — a real, honest possible outcome, not hidden


def calibration_stats() -> CalibrationStats:
    """Compute intra-person spread and inter-person distance across every
    registered individual's embeddings — the observed distributions
    DEFAULT_FACE_MATCH_DISTANCE_THRESHOLD should sit between, the same
    "calibrate against an observed distribution, don't guess" principle
    core/evaluation applies to the unsupervised image methods' anomaly
    threshold (autoencoder/PatchCore/Isolation Forest).

    Needs at least one individual with >=2 embeddings to say anything
    about intra-person spread, and at least two individuals (with >=1
    embedding each) to say anything about inter-person distance — with
    too little data for either, the corresponding list is empty and
    `suggested_threshold` is None rather than a number computed from
    nothing meaningful.
    """
    conn = get_connection(DB_PATH)
    try:
        _ensure_schema(conn)
        rows = conn.execute(f"SELECT * FROM {_EMBEDDINGS_TABLE}").fetchall()
    finally:
        conn.close()
    embeddings = [FaceEmbeddingRecord.from_row(r) for r in rows]

    by_person: dict[str, list[list[float]]] = {}
    for e in embeddings:
        by_person.setdefault(e.identity_key, []).append(e.embedding)

    intra: list[float] = []
    for vectors in by_person.values():
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                intra.append(_l2_distance(vectors[i], vectors[j]))

    inter: list[float] = []
    people = list(by_person.items())
    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            _, vectors_a = people[i]
            _, vectors_b = people[j]
            closest = min(_l2_distance(a, b) for a in vectors_a for b in vectors_b)
            inter.append(closest)

    if not intra or not inter:
        return CalibrationStats(intra_person_distances=intra, inter_person_distances=inter, suggested_threshold=None, separable=False)

    max_intra, min_inter = max(intra), min(inter)
    separable = max_intra < min_inter
    # Midpoint between the worst-case same-person distance and the
    # closest-case different-person distance — sits strictly between the
    # two observed distributions when they're actually separable; when
    # they're not (separable=False), this is still reported so a caller
    # can see exactly where the overlap is, not hidden behind None.
    suggested_threshold = (max_intra + min_inter) / 2
    return CalibrationStats(
        intra_person_distances=intra, inter_person_distances=inter,
        suggested_threshold=suggested_threshold, separable=separable,
    )
