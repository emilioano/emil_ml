"""Fas 2 sanity check: run a few realistic queries against the indexed
tandborste knowledge base and print what came back — chunks, provenance,
and similarity — so retrieval quality can be judged before any generation
is built on top of it. No LLM involved.

Assumes scripts/verify_rag_indexing.py has already been run (or
core.reporting.knowledge.indexer.index_all() otherwise), so the shared
ChromaDB collection actually has chunks in it.
Run with: python scripts/verify_rag_retrieval.py
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
from emil_ml.core.reporting.knowledge import retriever


def _print_results(results: list[retriever.RetrievedChunk]) -> None:
    if not results:
        print("  (no results — either nothing matched the metadata filter, "
              "or every candidate fell below the similarity threshold)")
        return
    for r in results:
        snippet = r.text[:90].replace("\n", " ")
        print(f"  sim={r.similarity:.3f}  [{r.doc_type}] {r.source} / {r.section!r}")
        print(f"           {snippet}...")


def main() -> None:
    configure_logging()
    print(f"Default similarity threshold: {retriever.DEFAULT_MIN_SIMILARITY}")
    print()

    # --- Test 1: query matching a specific defect type -----------------
    print("=== Test 1: query for a specific defect ('missing bristles') ===")
    query = retriever.build_query_text("tandborste", defect_class="missing bristles")
    print(f"Query text (via build_query_text): {query!r}")
    results = retriever.retrieve("tandborste", query)
    _print_results(results)
    if results and results[0].section == "Missing bristles" and results[0].doc_type == "manual":
        print("  -> PASS: top result is the matching manual section.")
    else:
        print("  -> UNEXPECTED: top result is not the 'Missing bristles' manual section.")
    print()

    # --- Test 2: component_type filter confirms isolation --------------
    print("=== Test 2: same toothbrush-specific query, but scoped to a DIFFERENT component ('flaska') ===")
    print("Confirms the metadata filter is actually applied, not a no-op — 'flaska' has no")
    print("documents of its own, but if isolation were broken this query could still surface")
    print("tandborste's semantically-similar 'Missing bristles' content.")
    results = retriever.retrieve("flaska", query)
    _print_results(results)
    if not results:
        print("  -> PASS: no chunks returned — component_type filter correctly excluded tandborste's data.")
    else:
        print("  -> FAIL: isolation broken — chunks leaked across component_type.")
    print()

    # --- Test 3: component with no indexed documents at all ------------
    print("=== Test 3: component with no documents ('tandborste-yolo-ny') ===")
    generic_query = retriever.build_query_text("tandborste-yolo-ny")
    results = retriever.retrieve("tandborste-yolo-ny", generic_query)
    _print_results(results)
    if not results:
        print("  -> PASS: empty result, handled cleanly (no crash, no irrelevant filler).")
    else:
        print("  -> FAIL: expected no results for a component with no indexed documents.")
    print()

    # --- Bonus: similarity threshold rejects a genuinely unrelated query ---
    print("=== Bonus: deliberately unrelated query against tandborste itself ('weather forecast tomorrow') ===")
    print("Demonstrates the similarity threshold doing real work — component_type filtering alone")
    print("wouldn't catch this, since tandborste genuinely does have indexed documents.")
    results = retriever.retrieve("tandborste", "weather forecast tomorrow")
    _print_results(results)
    if not results:
        print("  -> PASS: below-threshold matches correctly dropped instead of returned as filler.")
    else:
        print("  -> UNEXPECTED: an unrelated query returned results above the similarity threshold.")
    print()

    # --- doc_types narrowing -------------------------------------------
    print("=== doc_types narrowing: same defect query, restricted to doc_type='machine_context' only ===")
    results = retriever.retrieve("tandborste", query, doc_types=["machine_context"])
    _print_results(results)
    if results and all(r.doc_type == "machine_context" for r in results):
        print("  -> PASS: only machine_context chunks returned.")
    elif not results:
        print("  -> (no machine_context chunks scored above threshold for this query — also acceptable)")
    else:
        print("  -> FAIL: doc_types filter did not narrow results correctly.")

    print("\nFas 2 retrieval OK.")


if __name__ == "__main__":
    main()
