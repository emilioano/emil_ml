"""Fas 1 sanity check: index every registered component's knowledge/
directory into ChromaDB and list what landed there, so indexing can be
verified without needing retriever.py or an LLM yet.

Requires a local Ollama instance running with the embedding model pulled
(default: nomic-embed-text) — see core/reporting/knowledge/indexer.py.
Run with: python scripts/verify_rag_indexing.py
"""

from __future__ import annotations

import sys

# Windows terminals default to a non-UTF-8 codepage that mangles non-ASCII
# characters (°, em dashes, ...) on print() otherwise — the underlying
# stored text is unaffected either way, this only fixes how this script
# displays it.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from emil_ml.config.logging_config import configure_logging
from emil_ml.config.registry import ComponentRegistry
from emil_ml.config.settings import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR
from emil_ml.core.reporting.knowledge import indexer
from emil_ml.utils.paths import for_component


def main() -> None:
    configure_logging()
    print(f"ChromaDB persist dir: {CHROMA_PERSIST_DIR}")
    print()

    registry = ComponentRegistry()
    components = registry.list_all()
    print(f"Indexing knowledge/ for all {len(components)} registered component(s)...")
    counts = indexer.index_all()
    for name, count in counts.items():
        knowledge_dir = for_component(name).knowledge_dir
        marker = f"{count} chunk(s)" if count else "no documents"
        print(f"  {name}: {marker}  ({knowledge_dir})")
    print()

    indexed_components = {name: count for name, count in counts.items() if count > 0}
    if not indexed_components:
        print("No component had any documents under its knowledge/ directory — nothing to show below.")
        return

    collection = indexer.get_collection()
    total = collection.count()
    print(f"ChromaDB collection {CHROMA_COLLECTION_NAME!r} now has {total} chunk(s) total.")
    print()

    for component_name in sorted(indexed_components):
        result = collection.get(
            where={"component_type": component_name}, include=["documents", "metadatas"]
        )
        by_file: dict[str, list[tuple[dict, str]]] = {}
        for metadata, document in zip(result["metadatas"], result["documents"]):
            by_file.setdefault(metadata["file"], []).append((metadata, document))

        print(f"### component_type={component_name!r} — {len(result['ids'])} chunk(s), {len(by_file)} document(s)")
        for file, chunks in sorted(by_file.items()):
            first_meta = chunks[0][0]
            print(f"  === {file} (doc_type={first_meta['doc_type']!r}, source={first_meta['source']!r}) ===")
            for metadata, document in sorted(chunks, key=lambda c: (c[0]["section"], c[0]["chunk_index"])):
                section = metadata["section"] or "(no heading)"
                snippet = document[:100].replace("\n", " ")
                print(f"    [{section}] chunk {metadata['chunk_index']}: {snippet}...")
        print()

    print(f"Total: {len(indexed_components)} component(s) with documents, {total} chunk(s) in the shared collection.")
    print("\nFas 1 indexing OK (component-folder structure).")


if __name__ == "__main__":
    main()
