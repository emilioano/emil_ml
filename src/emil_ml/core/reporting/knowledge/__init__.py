"""Document knowledge base: indexing (indexer.py) and retrieval (retriever.py).

Documents live under each component's own knowledge/ subdirectory —
data/components/<name>/knowledge/*.md, see utils/paths.py's
ComponentPaths.knowledge_dir — alongside its training data and models, not
in a central directory. `indexer.py` gets the list of components to index
from the component registry, chunks and embeds their documents (via a
local Ollama instance) into one shared ChromaDB collection; `retriever.py`
filters by metadata (component_type first, always — see its own docstring
for why) before running similarity search over the filtered subset. That
metadata filter, not a physical split of the vector store, is what keeps
one component's documents from bleeding into another's retrieval results.
"""
