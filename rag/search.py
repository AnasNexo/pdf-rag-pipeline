"""Search primitives: BM25, dense semantic search, and hybrid RRF fusion.

All three functions return lists of plain dicts with the same keys
(chunk_id, score, text_content, chunk_type, source, metadata_json,
page_number, filename) so callers can treat them interchangeably.
"""

from __future__ import annotations

import re
import sqlite3

from config import DEFAULT_ALPHA, EMBEDDING_MODEL, RRF_K
from rag.embedder import encode_texts, unpack_vector

#: Common English function words removed from BM25 queries so natural-
#: language questions don't drown content terms in noise.
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "on", "at",
    "by", "for", "with", "about", "as", "from", "or", "and", "but", "not",
    "who", "what", "which", "that", "this", "these", "those", "it", "its",
    "he", "she", "they", "we", "you", "i", "me", "him", "her", "us", "them",
    "how", "when", "where", "why", "all", "any", "no", "so", "than", "just",
    "also", "more", "most", "some", "such", "into", "if", "my", "your",
    "his", "our", "their",
})


def _fts_escape(query: str) -> str:
    """Convert a natural-language query into a safe FTS5 MATCH expression.

    Content words (stopwords removed) are each wrapped in double quotes so
    FTS5 treats them literally (no accidental AND/OR/NOT/paren syntax),
    then joined with OR so BM25 can score chunks matching any subset of
    terms — chunks matching more terms still rank higher.

    Args:
        query (str): Raw user query.

    Returns:
        str: FTS5 MATCH expression (never empty; '""' as a last resort).
    """
    tokens = [t for t in re.findall(r"\w+", query) if t.lower() not in _STOPWORDS]
    if not tokens:
        tokens = re.findall(r"\w+", query)
    if not tokens:
        return '""'
    quoted = ['"' + t.replace('"', '""') + '"' for t in tokens]
    return " OR ".join(quoted)


def bm25_search(
    conn: sqlite3.Connection,
    query: str,
    top_k: int = 10,
    document_id: int | None = None,
) -> list[dict]:
    """Rank chunks against a query using FTS5's built-in BM25.

    FTS5's bm25() returns negative values (more negative = better match),
    so the inner query orders ascending. When a document filter is active
    the FTS limit is widened, since the filter is applied after ranking.

    Args:
        conn (sqlite3.Connection): Open database connection with a built
            FTS index.
        query (str): Natural-language query.
        top_k (int): Maximum results to return.
        document_id (int | None): Restrict results to one document.

    Returns:
        list[dict]: Result dicts ordered best-first.
    """
    doc_filter = "AND c.document_id = ?" if document_id is not None else ""
    params: list = [_fts_escape(query), top_k * (3 if document_id else 1)]
    if document_id is not None:
        params.append(document_id)
    cur = conn.execute(
        f"""
        SELECT
            fts_results.chunk_id,
            fts_results.score,
            c.text_content,
            c.chunk_type,
            c.source,
            c.metadata_json,
            p.page_number,
            d.filename
        FROM (
            SELECT rowid AS chunk_id, bm25(chunks_fts) AS score
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
        ) fts_results
        JOIN chunks c ON c.id = fts_results.chunk_id
        LEFT JOIN pages p ON p.id = c.page_id
        LEFT JOIN documents d ON d.id = c.document_id
        WHERE 1=1 {doc_filter}
        """,
        params,
    )
    return [dict(r) for r in cur.fetchall()]


def semantic_search(
    conn: sqlite3.Connection,
    query: str,
    top_k: int = 10,
    model_name: str = EMBEDDING_MODEL,
    document_id: int | None = None,
) -> list[dict]:
    """Rank chunks against a query by embedding cosine similarity.

    Stored vectors are L2-normalised at embed time, so a dot product with
    the (also normalised) query vector equals cosine similarity.

    Args:
        conn (sqlite3.Connection): Open database connection with built
            embeddings.
        query (str): Natural-language query.
        top_k (int): Maximum results to return.
        model_name (str): Must match the model used at index time.
        document_id (int | None): Restrict results to one document.

    Returns:
        list[dict]: Result dicts ordered best-first.
    """
    import numpy as np

    q_vec = encode_texts([query], model_name)[0]  # shape (dim,)

    doc_filter = "AND c.document_id = ?" if document_id is not None else ""
    rows = conn.execute(
        f"""
        SELECT ce.chunk_id, ce.embedding,
               c.text_content, c.chunk_type, c.source, c.metadata_json,
               p.page_number, d.filename
        FROM chunk_embeddings ce
        JOIN chunks c ON c.id = ce.chunk_id
        LEFT JOIN pages p ON p.id = c.page_id
        LEFT JOIN documents d ON d.id = c.document_id
        WHERE 1=1 {doc_filter}
        """,
        (document_id,) if document_id is not None else (),
    ).fetchall()

    if not rows:
        return []

    matrix = np.array([unpack_vector(r["embedding"]) for r in rows], dtype=np.float32)
    scores = matrix @ q_vec  # (n,) cosine similarities

    top_idx = scores.argsort()[::-1][:top_k]
    return [
        {
            "chunk_id": rows[i]["chunk_id"],
            "score": float(scores[i]),
            "text_content": rows[i]["text_content"],
            "chunk_type": rows[i]["chunk_type"],
            "source": rows[i]["source"],
            "metadata_json": rows[i]["metadata_json"],
            "page_number": rows[i]["page_number"],
            "filename": rows[i]["filename"],
        }
        for i in top_idx
    ]


def hybrid_search(
    conn: sqlite3.Connection,
    query: str,
    top_k: int = 10,
    model_name: str = EMBEDDING_MODEL,
    alpha: float = DEFAULT_ALPHA,
    document_id: int | None = None,
    guaranteed_semantic: int = 0,
) -> list[dict]:
    """Combine BM25 and semantic rankings via Reciprocal Rank Fusion.

    RRF penalises a chunk that one leg misses entirely (it gets a fallback
    rank), so a chunk ranked highly by semantic search but absent from BM25
    can be pushed out of the top_k — even though it may be the true answer.
    ``guaranteed_semantic`` protects against this: the top-N pure-semantic
    chunks are always included in the returned pool regardless of their RRF
    score. They are appended after the RRF-ranked results, so RRF still
    decides the ordering of everything it surfaced; this just guarantees a
    strong semantic hit reaches a downstream reranker that can score it
    properly. Use it when the caller reranks (otherwise it only pads the tail).

    Args:
        conn (sqlite3.Connection): Open database connection with both
            indexes built.
        query (str): Natural-language query.
        top_k (int): Maximum results to return.
        model_name (str): Embedding model for the semantic leg.
        alpha (float): Blend weight — 0 is pure BM25 (exact terms),
            1 is pure semantic (paraphrase/concept), 0.5 is balanced.
        document_id (int | None): Restrict results to one document.
        guaranteed_semantic (int): Force-include this many top pure-semantic
            chunks into the result even if RRF would drop them. 0 disables.

    Returns:
        list[dict]: Result dicts ordered best-first, with RRF scores.
    """
    bm25 = bm25_search(conn, query, top_k=top_k * 2, document_id=document_id)
    sem = semantic_search(conn, query, top_k=top_k * 2,
                          model_name=model_name, document_id=document_id)

    bm25_rank = {r["chunk_id"]: i + 1 for i, r in enumerate(bm25)}
    sem_rank = {r["chunk_id"]: i + 1 for i, r in enumerate(sem)}

    # Pool of all candidate chunks with their metadata.
    pool: dict[int, dict] = {}
    for r in bm25 + sem:
        pool.setdefault(r["chunk_id"], r)

    fallback_rank = top_k * 2 + 1  # rank for chunks absent from one list

    rrf_scores = {
        cid: (1 - alpha) / (RRF_K + bm25_rank.get(cid, fallback_rank))
             + alpha / (RRF_K + sem_rank.get(cid, fallback_rank))
        for cid in pool
    }

    ranked = sorted(pool, key=lambda c: rrf_scores[c], reverse=True)[:top_k]

    # Guarantee the strongest semantic hits survive into the pool, appending
    # any that RRF dropped so a downstream reranker can still see them.
    if guaranteed_semantic > 0:
        present = set(ranked)
        for r in sem[:guaranteed_semantic]:
            cid = r["chunk_id"]
            if cid not in present:
                ranked.append(cid)
                present.add(cid)

    return [{**pool[c], "score": rrf_scores[c]} for c in ranked]
