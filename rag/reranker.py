"""Cross-encoder reranking of first-stage retrieval candidates.

First-stage search (BM25 / dense / hybrid) ranks chunks cheaply but
approximately: BM25 ignores meaning, and the bi-encoder embeds the query and
each chunk independently, so it never compares them jointly. A cross-encoder
instead feeds the (query, chunk) pair through a single transformer and emits a
direct relevance score — slower, but markedly more precise at the top of the
list.

The reranker therefore runs as a second stage: take the top-N candidates from
first-stage search, rescore every (query, chunk) pair, and reorder. The model
is cached like the embedding model in rag.embedder.
"""

from __future__ import annotations

from config import RERANK_MODEL

_model_cache: dict[str, "CrossEncoder"] = {}


def get_reranker(model_name: str = RERANK_MODEL):
    """Return a cached CrossEncoder, loading it on first use.

    Args:
        model_name (str): Hugging Face cross-encoder model name.

    Returns:
        CrossEncoder: The cached model instance.
    """
    if model_name not in _model_cache:
        from sentence_transformers import CrossEncoder
        print(f"  Loading reranker '{model_name}'...")
        _model_cache[model_name] = CrossEncoder(model_name)
    return _model_cache[model_name]


def rerank(
    query: str,
    candidates: list,
    text_getter,
    top_k: int,
    model_name: str = RERANK_MODEL,
) -> list:
    """Reorder candidates by cross-encoder relevance and keep the top_k.

    The cross-encoder score overwrites nothing on the candidate objects; it is
    used purely for ordering, so callers keep their original score field.

    Args:
        query (str): The user query.
        candidates (list): First-stage candidates, any object type.
        text_getter (Callable): Maps a candidate to the text to score against
            the query (e.g. lambda c: c.text_content).
        top_k (int): Number of candidates to return after reranking.
        model_name (str): Cross-encoder model name.

    Returns:
        list: Up to top_k candidates, most relevant first. Returns the input
            unchanged (truncated to top_k) when there is nothing to rerank.
    """
    if not candidates:
        return []
    if len(candidates) == 1:
        return candidates[:top_k]

    model = get_reranker(model_name)
    pairs = [(query, text_getter(c)) for c in candidates]
    scores = model.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda cs: cs[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]
