"""Optional reranking interface for retrieved chunks.

TODO (Future Phase): Implement BaseReranker interface and optional local cross-encoder model:
- Re-score candidate chunks based on query-passage cross-attention
- Re-sort candidate chunks before prompt construction
"""
