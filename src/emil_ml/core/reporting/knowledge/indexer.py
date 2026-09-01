"""Chunks and embeds a component's knowledge-base documents into ChromaDB.

Documents live under that component's own knowledge/ subdirectory —
data/components/<name>/knowledge/*.md (see utils/paths.py's
ComponentPaths.knowledge_dir) — alongside its training data and models,
not in a central directory. `index_all()` gets the list of components to
index from the component registry, the same source of truth every other
per-component operation in this app uses, rather than scanning a
directory itself. Each file starts with a small YAML frontmatter block,
then markdown headings that become each chunk's "section" metadata:

    ---
    doc_type: manual
    source: Toothbrush Maintenance Manual v1
    ---
    # Typical defects

    ...text...

    ## Missing bristles
    ...text...

Chunking is word-count-based with overlap (RAG_CHUNK_SIZE_WORDS /
RAG_CHUNK_OVERLAP_WORDS in config/settings.py) — simple and predictable for
these short spec-sheet-style documents, not a semantic chunker.

Embeddings come from a local Ollama instance (OLLAMA_EMBEDDINGS_URL), not a
Python client library — chromadb is used purely as the vector store
(storage + metadata filtering), never for its own default embedding
function, since a query later has to be embedded with the exact same model
the chunks were indexed with (see retriever.py). Every component's chunks
share one ChromaDB collection; a chunk's `component_type` metadata field
(the component's registry `name`) is what isolates one component's
documents from another's at query time — retriever.py filters on it
before it ever runs a similarity search, not a physical split of the
vector store.

Re-running index_component_type() is idempotent: chunk ids are
deterministic (component name + filename + section + position), so
ChromaDB's upsert overwrites a chunk in place rather than duplicating it —
editing a document and re-indexing just replaces its old chunks.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import yaml

from emil_ml.config.registry import ComponentRegistry
from emil_ml.config.settings import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    DEFAULT_RAG_EMBEDDING_MODEL,
    OLLAMA_EMBEDDINGS_URL,
    OLLAMA_MAX_RETRIES,
    OLLAMA_RETRY_BACKOFF_SECONDS,
    RAG_CHUNK_OVERLAP_WORDS,
    RAG_CHUNK_SIZE_WORDS,
    VALID_DOC_TYPES,
)
from emil_ml.utils.paths import for_component

logger = logging.getLogger(__name__)

_DOC_EXTENSIONS = {".md", ".txt"}
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    """One indexed piece of text, with enough provenance to cite it in a report."""

    id: str
    text: str
    component_type: str  # the owning component's registry `name`
    doc_type: str
    source: str
    section: str
    chunk_index: int
    file: str  # filename within that component's knowledge/ directory

    def metadata(self) -> dict[str, Any]:
        return {
            "component_type": self.component_type,
            "doc_type": self.doc_type,
            "source": self.source,
            "section": self.section,
            "chunk_index": self.chunk_index,
            "file": self.file,
        }


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a leading '---\\n...\\n---' YAML block off the document body."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    meta = yaml.safe_load(text[3:end].strip()) or {}
    body = text[end + 4 :].lstrip("\n")
    return meta, body


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Split a document body into (heading, text) pairs at markdown headings.

    A document with no headings becomes a single section with an empty
    heading — still indexed, just without a section label.
    """
    matches = list(_HEADING_RE.finditer(body))
    if not matches:
        stripped = body.strip()
        return [("", stripped)] if stripped else []
    sections = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = body[start:end].strip()
        if text:
            sections.append((m.group(1).strip(), text))
    return sections


def _chunk_words(text: str, *, size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks of ~`size` words each."""
    words = text.split()
    if not words:
        return []
    step = max(size - overlap, 1)
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks


def parse_document(path: Path, component_name: str) -> list[Chunk]:
    """Parse one document into its (section, chunk) pieces — no embedding/storage yet.

    Split out from index_component_type() so it's independently testable
    and inspectable (e.g. from the verification script) without needing
    Ollama or ChromaDB running.
    """
    raw = path.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(raw)

    doc_type = meta.get("doc_type", "manual")
    if doc_type not in VALID_DOC_TYPES:
        raise ValueError(f"{path}: doc_type {doc_type!r} must be one of {VALID_DOC_TYPES}")
    source = meta.get("source", path.stem)

    chunks: list[Chunk] = []
    for section_title, section_text in _split_sections(body):
        section_slug = re.sub(r"[^a-z0-9]+", "-", section_title.lower()).strip("-") or "root"
        for i, chunk_text in enumerate(
            _chunk_words(section_text, size=RAG_CHUNK_SIZE_WORDS, overlap=RAG_CHUNK_OVERLAP_WORDS)
        ):
            chunks.append(
                Chunk(
                    id=f"{component_name}:{path.stem}:{section_slug}:{i}",
                    text=chunk_text,
                    component_type=component_name,
                    doc_type=doc_type,
                    source=source,
                    section=section_title,
                    chunk_index=i,
                    file=path.name,
                )
            )
    return chunks


def embed(text: str, *, model: str = DEFAULT_RAG_EMBEDDING_MODEL) -> list[float]:
    """Embed a single piece of text via Ollama.

    Shared by indexer.py (embedding chunks) and retriever.py (embedding a
    query) — a query must be embedded with the same model its knowledge
    base was indexed with, since different models' embedding spaces aren't
    comparable.

    Retries on failure (see OLLAMA_MAX_RETRIES in config/settings.py): a
    cold model load — the first request after Ollama's own idle timeout
    unloads it — can trip a GPU-discovery watchdog on Ollama's side and
    return a transient 500, even though the very next request succeeds.
    Indexing a document loops this call once per chunk with no
    higher-level retry, so without this the whole batch would abort on
    that first chunk.
    """
    last_exc: requests.exceptions.RequestException | None = None
    for attempt in range(OLLAMA_MAX_RETRIES):
        try:
            response = requests.post(OLLAMA_EMBEDDINGS_URL, json={"model": model, "prompt": text}, timeout=60)
            response.raise_for_status()
            return response.json()["embedding"]
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt + 1 < OLLAMA_MAX_RETRIES:
                logger.warning(
                    "Ollama embedding request failed (attempt %d/%d): %s — retrying",
                    attempt + 1, OLLAMA_MAX_RETRIES, exc,
                )
                time.sleep(OLLAMA_RETRY_BACKOFF_SECONDS * (attempt + 1))
    assert last_exc is not None
    logger.error("Ollama embedding request failed after %d attempt(s): %s", OLLAMA_MAX_RETRIES, last_exc)
    raise last_exc


def get_collection():  # noqa: ANN201 - chromadb.Collection, imported lazily
    """The shared ChromaDB collection every component's knowledge-base chunks live in.

    Public (not module-private) because retriever.py needs the exact same
    collection to query against. Imported inside the function, not at
    module top level, so nothing outside core/reporting/knowledge/ ever
    requires the `rag` extra to be installed — same reasoning as
    core/anomaly/patchcore/*.py's lazy anomalib imports.
    """
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    # Cosine, not ChromaDB's l2 (Euclidean) default: nomic-embed-text
    # vectors aren't unit-normalized, so squared L2 distance blows up to
    # a large, model-specific scale that's unusable for a fixed similarity
    # threshold. Cosine distance is bounded ([0, 2], i.e. 1 - cosine
    # similarity) regardless of vector magnitude — see retriever.py's
    # RetrievedChunk.similarity, which is exactly `1 - distance` here.
    return client.get_or_create_collection(name=CHROMA_COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def _delete_stale_chunks(collection, component_name: str, current_chunk_ids: set[str]) -> int:  # noqa: ANN001
    """Remove this component's indexed chunks that this indexing run didn't
    just (re-)produce — i.e. reconcile against `current_chunk_ids`, not
    against which *filenames* are present.

    upsert() (above/below) only ever adds/overwrites chunks for whatever
    this run parsed — nothing else in this module calls delete(), so
    anything indexed previously that this run didn't reproduce would
    otherwise stay in ChromaDB forever, silently citable in a report.

    Originally this compared against filenames still on disk, which
    misses a real case: a file that still EXISTS under the same name but
    whose CONTENT changed (different section headings, e.g. a document
    edited or accidentally replaced with different content) produces
    different chunk ids (id = component:filename:section-slug:index) —
    the old section's chunks were never "removed" by filename, just
    silently orphaned under stale ids. Confirmed happening in practice:
    a knowledge-base file was overwritten with different content, and
    the previous content's chunks stayed indexed and got retrieved
    (and cited in a report) indefinitely, since their filename was still
    "current". Comparing against the exact set of ids this run produced
    catches both cases with the same check — a removed file's chunks and
    a changed file's stale-section chunks are both just "ids that exist
    in the collection but weren't in this run's output."
    """
    existing = collection.get(where={"component_type": component_name}, include=[])
    stale_ids = [chunk_id for chunk_id in existing["ids"] if chunk_id not in current_chunk_ids]
    if stale_ids:
        collection.delete(ids=stale_ids)
    return len(stale_ids)


def count_component_chunks(component_name: str) -> int:
    """How many chunks this component currently has indexed — a
    read-only count, for core/component_deletion.py's deletion-impact
    summary (so a permanent-delete confirmation can show "N knowledge
    chunks" without actually touching anything).
    """
    collection = get_collection()
    existing = collection.get(where={"component_type": component_name}, include=[])
    return len(existing["ids"])


def delete_component_chunks(component_name: str) -> int:
    """Remove EVERY indexed chunk for this component from ChromaDB —
    unlike _delete_stale_chunks() above (which only removes what one
    indexing run's own reconciliation made obsolete), this unconditionally
    wipes the component's entire footprint in the shared collection. The
    one place a component's ChromaDB presence is fully erased, used
    exclusively by core/component_deletion.py's
    permanently_delete_component() — so a new component created later
    with the same slug never inherits a predecessor's stale chunks,
    exactly the cross-component leak family this project has already hit
    once (see retriever.py's isolation-violation warning). Idempotent:
    calling this on a component with 0 chunks left (e.g. a resumed,
    partially-completed deletion) is a safe no-op.
    """
    collection = get_collection()
    existing = collection.get(where={"component_type": component_name}, include=[])
    ids = existing["ids"]
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def index_component_type(component_name: str, *, embedding_model: str = DEFAULT_RAG_EMBEDDING_MODEL) -> int:
    """Index every document under this component's knowledge/ directory into ChromaDB.

    Returns the number of chunks (re-)indexed. A component with no
    knowledge/ directory yet (or an empty one) indexes 0 chunks rather
    than raising — most components won't have documentation, and that's a
    normal, unremarkable state, not an error (see `create_all()` in
    utils/paths.py, which creates this directory for every component
    regardless of whether it's ever populated). Either way, any
    previously-indexed chunks for files no longer present are still
    cleaned up (see _delete_stale_chunks) — even a component with zero
    current documents gets fully reconciled, not just early-returned past.
    """
    docs_dir = for_component(component_name).knowledge_dir
    paths = (
        sorted(p for p in docs_dir.iterdir() if p.is_file() and p.suffix.lower() in _DOC_EXTENSIONS)
        if docs_dir.exists()
        else []
    )

    all_chunks: list[Chunk] = []
    for path in paths:
        chunks = parse_document(path, component_name)
        logger.debug("component=%s %s -> %d chunk(s)", component_name, path.name, len(chunks))
        all_chunks.extend(chunks)

    collection = get_collection()
    if all_chunks:
        embeddings = [embed(c.text, model=embedding_model) for c in all_chunks]
        collection.upsert(
            ids=[c.id for c in all_chunks],
            embeddings=embeddings,
            documents=[c.text for c in all_chunks],
            metadatas=[c.metadata() for c in all_chunks],
        )
    stale_count = _delete_stale_chunks(collection, component_name, {c.id for c in all_chunks})
    if stale_count:
        logger.info("component=%s removed %d stale chunk(s) no longer produced by knowledge/", component_name, stale_count)
    logger.info("component=%s indexed %d chunk(s) from %d document(s)", component_name, len(all_chunks), len(paths))
    return len(all_chunks)


def index_all(*, embedding_model: str = DEFAULT_RAG_EMBEDDING_MODEL) -> dict[str, int]:
    """Index every ACTIVE registered component's knowledge/ directory —
    but only components with reporting_enabled on.

    Reporting is opt-in per component (see core/reporting/reporter.py);
    a component with it off is never touched here, matching
    should_generate_report() gating everything on the read side. Indexing
    a disabled component wouldn't be wrong exactly, just pointless work
    (and a stale index nothing ever queries) — reporting_enabled is what
    a caller actually meant to opt into. Components with reporting on but
    no documents still index 0 chunks and are included in the returned
    dict, same as before.

    Uses list_active() (lifecycle_status='active'), not list_all() — a
    deactivated or soft-deleted component is excluded from active-use
    iteration everywhere, RAG reindexing included, even if it still has
    reporting_enabled on. A soft-deleted component's chunks are left
    exactly as they were (untouched, matching soft-delete's "zero data
    impact" guarantee) — only permanently deleting it actually removes
    them (see core/component_deletion.py's delete_component_chunks()).
    """
    registry = ComponentRegistry()
    results: dict[str, int] = {}
    for component in registry.list_active():
        if not component.reporting_enabled:
            continue
        results[component.name] = index_component_type(component.name, embedding_model=embedding_model)
    return results
