"""Filtered similarity search over the shared knowledge-base ChromaDB collection.

Read-only — the retrieval half of core/reporting/knowledge/ (see
indexer.py for how chunks got there in the first place). No LLM involved:
this module's whole job is "given a situation, which documented chunks
are relevant", independently testable and inspectable before any report
text is ever generated (see scripts/verify_rag_retrieval.py).

Metadata filtering happens BEFORE similarity search, never after:
`component_type` is always part of the filter passed to ChromaDB, so a
query for one component can never surface another component's chunks no
matter how similar their embeddings happen to be — that's the isolation
guarantee retrieval provides. See build_where() for the one place
ChromaDB's filter syntax is dealt with; Fas 3's machine-context
conditions (e.g. "over-temperature") plug into that same function without
touching anything else here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from emil_ml.config.settings import DEFAULT_RAG_EMBEDDING_MODEL
from emil_ml.core.reporting.knowledge.indexer import embed, get_collection

logger = logging.getLogger(__name__)

DEFAULT_N_RESULTS = 5

# Compared against `similarity` (see RetrievedChunk) — cosine similarity,
# bounded to [-1, 1] — not raw ChromaDB distance. Calibrated against this
# project's own indexed data (scripts/verify_rag_retrieval.py): a query
# matching real content scored 0.57-0.74 across its relevant chunks; a
# deliberately unrelated query ("weather forecast tomorrow") topped out at
# 0.485 — its best "match" was still just shared domain vocabulary, not
# genuine relevance. All of this knowledge base's chunks share enough
# manufacturing/defect vocabulary that similarity scores sit higher
# overall than they would over a broader, more varied corpus; 0.5 is
# comfortably above the observed irrelevant-query ceiling and below the
# observed relevant-query floor for this data, not a generic embedding-
# space constant — re-calibrate if the knowledge base's character changes
# substantially (e.g. adding very different component types).
DEFAULT_MIN_SIMILARITY = 0.5

# The absolute floor above is a blunt instrument in this knowledge base:
# every chunk shares enough manufacturing/defect vocabulary that even
# clearly-unrelated sections of the SAME document sit within a few
# hundredths of the genuinely relevant one — e.g. a "missing_straws" query
# against the toothbrush manual scored "Missing bristles" (the right
# section) at 0.60, but "Sharp edges / flash residue" (a different
# defect) at 0.595, "Discoloration" at 0.588, and "Typical defects" at
# 0.566 all cleared the 0.5 floor too and were returned alongside it. A
# relative margin fixes that: keep a chunk only if it's within
# `relative_margin` of the best-scoring chunk, not just above a fixed
# floor.
#
# That margin is anchored per doc_type's own top match in the candidate
# pool, not the single best match across all of them — confirmed directly
# (see scripts/verify_rag_retrieval.py) that anchoring to one global top
# score breaks the case that matters most: when a detected defect
# coincides with a genuinely matching machine-context anomaly, the
# anomaly's exact-term match can dominate the query (0.71 for "Elevated
# vibration" vs. 0.59 for the defect's own "Missing bristles" section — a
# 0.12 gap) and a global-top margin would silently drop the defect's own
# documentation right when it's most needed. Anchoring per doc_type means
# "manual"'s own best match and "machine_context"'s own best match are
# each judged against their own category's top, so one category's
# dominant hit can never suppress another's.
#
# 0.006 is calibrated tight against this KB's real clustering, not a
# generic constant — see scripts/verify_rag_retrieval.py for the exact
# before/after, and retrieve_for_inspection()'s docstring for why it's
# safe to keep this tight (that function's per-signal sub-queries mean a
# chunk that's the actual subject of a query is always that query's own
# rank-1 match, trivially surviving any margin value — tightening this
# constant can no longer accidentally drop the chunk a query was
# specifically about, the way it could before that split existed). At
# 0.006, a "missing_straws" query against the toothbrush manual keeps only
# "Missing bristles" (the right section) among that doc_type's chunks —
# "Sharp edges / flash residue" (0.006 below it) and "Discoloration"/
# "Typical defects" (further still) are all correctly dropped as noise.
# One structural residual: a doc_type with only ONE chunk in the
# candidate pool always survives regardless of margin (it's trivially its
# own top, zero gap) — e.g. a single incident-report chunk may still
# accompany a tight manual-section match. That's an accepted tradeoff,
# not a bug: distinguishing "the sole candidate in an otherwise-absent
# category" from "genuinely relevant" would need a different mechanism
# than a similarity margin. Re-calibrate alongside DEFAULT_MIN_SIMILARITY
# if the knowledge base's character changes substantially.
DEFAULT_RELATIVE_MARGIN = 0.006


def build_where(conditions: dict[str, Any]) -> dict[str, Any] | None:
    """Build a ChromaDB `where` filter from a flat dict of conditions.

    Encapsulates a ChromaDB 1.x quirk: a `where` dict with more than one
    top-level key is rejected ("Expected where to have exactly one
    operator, got {...}") — multiple conditions must be wrapped in an
    explicit `$and`. Every filter this module builds goes through this
    one function, so that syntax lives in exactly one place rather than
    being repeated at every call site.

    `None`-valued conditions are dropped (lets callers pass optional
    filters straight through without an `if` at every call site). Returns
    `None` — ChromaDB's "no filter" value — if nothing is left.
    """
    conditions = {key: value for key, value in conditions.items() if value is not None}
    if not conditions:
        return None
    if len(conditions) == 1:
        ((key, value),) = conditions.items()
        return {key: value}
    return {"$and": [{key: value} for key, value in conditions.items()]}


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved chunk, with the provenance a report cites it by."""

    text: str
    component_type: str
    doc_type: str
    source: str
    section: str
    file: str  # filename within component_type's knowledge/ directory (see indexer.Chunk.file)
    distance: float  # raw ChromaDB cosine distance (1 - cosine similarity); lower = more similar
    similarity: float  # cosine similarity, 1 - distance, in [-1, 1]; higher = more similar


def build_query_text(
    component_type: str,
    *,
    defect_class: str | None = None,
    machine_states: Sequence[str] | None = None,
    intent: str = "cause and corrective action",
) -> str:
    """Deterministic query string built from structured detection fields.

    In the full pipeline (Fas 4), the query comes from a PredictionResult
    plus machine_context.analyzer's searchable states (Fas 3) — not a
    person typing a question — so this stays simple and predictable (the
    same inputs always produce the exact same string) rather than clever,
    so a bad retrieval result is easy to debug: read the query back and
    it's obvious what was asked.

    `machine_states` matters most exactly when `defect_class` is absent
    (PatchCore/autoencoder give no defect label) — the machine's own
    anomalies (e.g. "over-temperature") may be the only concrete lead the
    query has to offer.

    Called with at most one of `defect_class`/`machine_states` set per
    call from retrieve_for_inspection() (one sub-query per signal, not
    both folded into a single string) — see its own docstring for why.
    Still accepts both at once here, since this function is just string
    assembly and other callers (tests, scripts) may want one combined
    query text for inspection/debugging.
    """
    parts = [component_type]
    if defect_class:
        parts.append(defect_class)
    if machine_states:
        parts.extend(machine_states)
    parts.append(intent)
    return " ".join(parts)


def retrieve(
    component_type: str,
    query_text: str,
    *,
    doc_types: Sequence[str] | None = None,
    n_results: int = DEFAULT_N_RESULTS,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    relative_margin: float = DEFAULT_RELATIVE_MARGIN,
    embedding_model: str = DEFAULT_RAG_EMBEDDING_MODEL,
) -> list[RetrievedChunk]:
    """Retrieve the most relevant knowledge-base chunks for `component_type`.

    `component_type` is always applied as a metadata filter before any
    similarity search runs — never optional, never applied after the
    fact. `doc_types`, if given, narrows further (e.g. only "manual" and
    "spec", excluding "incident"); Fas 3 will add machine-context
    conditions the same way, through build_where().

    Two relevance filters apply, in order:
    1. `min_similarity` — the absolute floor. Chunks below it are dropped
       rather than kept just to fill `n_results`: an empty list is a
       real, meaningful result here — "nothing relevant was found" — not
       a failure to signal downstream (Fas 4's report generation must say
       so rather than inventing an answer from irrelevant chunks).
    2. `relative_margin` — see DEFAULT_RELATIVE_MARGIN's own comment for
       why this is needed on top of the floor, and why it's anchored per
       doc_type's own top match within the surviving candidates rather
       than the single overall best match: a chunk is dropped if it
       trails its own doc_type's best-scoring surviving chunk by more
       than this margin, so a dominant match in one doc_type (e.g. a
       machine-context anomaly whose exact wording is in the query) can
       never suppress a genuinely relevant match in another (e.g. the
       defect's own manual section).
    """
    conditions: dict[str, Any] = {"component_type": component_type}
    if doc_types:
        conditions["doc_type"] = {"$in": list(doc_types)}
    where = build_where(conditions)
    logger.debug("component=%s query=%r where=%s n_results=%d", component_type, query_text, where, n_results)

    query_embedding = embed(query_text, model=embedding_model)
    results = get_collection().query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results["distances"] else []

    # Absolute floor first, same as before.
    candidates = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        similarity = 1.0 - distance
        if similarity < min_similarity:
            continue
        candidates.append((document, metadata, distance, similarity))

    # Then the relative margin, anchored per doc_type's own top surviving
    # match (see DEFAULT_RELATIVE_MARGIN's comment above).
    top_similarity_by_doc_type: dict[str, float] = {}
    for _, metadata, _, similarity in candidates:
        doc_type = metadata["doc_type"]
        if similarity > top_similarity_by_doc_type.get(doc_type, float("-inf")):
            top_similarity_by_doc_type[doc_type] = similarity

    chunks: list[RetrievedChunk] = []
    for document, metadata, distance, similarity in candidates:
        doc_type_top = top_similarity_by_doc_type[metadata["doc_type"]]
        if similarity < doc_type_top - relative_margin:
            continue
        # A chunk's own component_type disagreeing with the component_type
        # this query filtered on should be structurally impossible — the
        # `where` filter above already restricts ChromaDB's search to it —
        # so seeing one here means the filter didn't do its job (or the
        # data itself is mistagged) and isolation between components has
        # broken down. Confirmed directly: this is exactly the shape a
        # real cross-component leak took (a transistor incident report
        # retrieved for a toothbrush query) — worth a WARNING regardless
        # of cause, since by definition it's an isolation violation, not
        # a relevance-tuning question.
        if metadata["component_type"] != component_type:
            logger.warning(
                "component=%s retrieved chunk with component_type=%s (source=%r section=%r) — isolation violation",
                component_type, metadata["component_type"], metadata["source"], metadata["section"],
            )
        chunks.append(
            RetrievedChunk(
                text=document,
                component_type=metadata["component_type"],
                doc_type=metadata["doc_type"],
                source=metadata["source"],
                section=metadata["section"],
                file=metadata["file"],
                distance=distance,
                similarity=similarity,
            )
        )
    logger.debug(
        "component=%s retrieved %d chunk(s): %s",
        component_type,
        len(chunks),
        [(c.doc_type, c.source, c.section, round(c.similarity, 4)) for c in chunks],
    )
    return chunks


def retrieve_for_inspection(
    component_type: str,
    *,
    defect_class: str | None = None,
    machine_states: Sequence[str] | None = None,
    doc_types: Sequence[str] | None = None,
    n_results: int = DEFAULT_N_RESULTS,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    relative_margin: float = DEFAULT_RELATIVE_MARGIN,
    embedding_model: str = DEFAULT_RAG_EMBEDDING_MODEL,
) -> list[RetrievedChunk]:
    """Retrieve chunks for one inspection's defect class AND machine
    anomalies as SEPARATE queries, then merge — not one query with
    everything folded into a single string.

    A single combined query text (component + defect_class + all
    machine_states + intent) has a real failure mode, confirmed directly
    against this project's own indexed data
    (scripts/verify_reporting_generation.py's Scenario 1 caught it): when
    a component has more than one machine-context document (e.g. separate
    sections for "Over-temperature" and "Elevated vibration"), folding
    both the defect term and the anomaly term into one query can make an
    UNRELATED same-doc_type section outscore the one that's actually
    correct — e.g. "Elevated vibration"'s text happens to mention "missing
    bristles" prominently, so it out-competed "Over-temperature" for a
    query about a defect class ("missing bristles") + an anomaly that was
    actually measured ("over-temperature"), even though vibration was
    never anomalous for that inspection. retrieve()'s per-doc_type margin
    only protects against one doc_type's dominant match suppressing
    ANOTHER doc_type's — it can't tell two same-doc_type documents apart
    when they're both pulled toward the query by different words in it.
    A dedicated single-topic query per signal doesn't have that ambiguity
    to begin with: "component + missing bristles + intent" and "component
    + over-temperature + intent" each cleanly surface their own correct
    top match, confirmed against the same data.

    Chunks are deduplicated by (source, section) across sub-queries,
    keeping first occurrence (defect_class's sub-query runs first, so a
    chunk relevant to both keeps its defect-query score) — order within
    each sub-query is otherwise preserved, unaffected by merging.

    No defect_class and no machine_states (PatchCore/autoencoder with no
    anomalies detected at all) falls back to a single plain
    component-level query — the same shape retrieve() always had for
    that case.
    """
    queries: list[str] = []
    if defect_class:
        queries.append(build_query_text(component_type, defect_class=defect_class))
    for state in machine_states or []:
        queries.append(build_query_text(component_type, machine_states=[state]))
    if not queries:
        queries.append(build_query_text(component_type))

    seen: set[tuple[str, str]] = set()
    merged: list[RetrievedChunk] = []
    for query_text in queries:
        for chunk in retrieve(
            component_type,
            query_text,
            doc_types=doc_types,
            n_results=n_results,
            min_similarity=min_similarity,
            relative_margin=relative_margin,
            embedding_model=embedding_model,
        ):
            key = (chunk.source, chunk.section)
            if key in seen:
                continue
            seen.add(key)
            merged.append(chunk)
    return merged
