"""Search-index construction: BM25 (SQLite FTS5) and dense embeddings.

Both builders are incremental and idempotent — they only index chunks
that are not yet present in their respective index tables.
"""

from __future__ import annotations

import sqlite3

from config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL
from rag.db import ensure_embedding_table, ensure_fts_table
from rag.embedder import encode_texts, pack_vector


def build_fts_index(conn: sqlite3.Connection) -> int:
    """Populate the FTS5 BM25 index with any not-yet-indexed chunks.

    The FTS rowid mirrors chunks.id so the two tables join directly.

    Args:
        conn (sqlite3.Connection): Open database connection with a
            populated chunks table.

    Returns:
        int: Total number of indexed rows after the operation.
    """
    ensure_fts_table(conn)
    conn.execute("""
        INSERT INTO chunks_fts(rowid, text_content)
        SELECT c.id, c.text_content
        FROM chunks c
        WHERE c.id NOT IN (SELECT rowid FROM chunks_fts)
    """)
    count = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    conn.commit()
    return count


def build_embeddings(
    conn: sqlite3.Connection,
    model_name: str = EMBEDDING_MODEL,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    reembed: bool = False,
) -> int:
    """Embed all chunks that do not yet have a stored vector.

    Each embedding records the model that produced it. When ``reembed`` is
    set, any embedding made by a *different* model is deleted first, so
    switching EMBEDDING_MODEL re-indexes the whole corpus consistently
    (mixing vectors from different models would make cosine scores
    meaningless).

    Args:
        conn (sqlite3.Connection): Open database connection with a
            populated chunks table.
        model_name (str): sentence-transformers model to use.
        batch_size (int): Chunks encoded per batch.
        reembed (bool): Drop embeddings from other models before building,
            forcing a full re-embed under ``model_name``.

    Returns:
        int: Number of newly embedded chunks.
    """
    ensure_embedding_table(conn)

    if reembed:
        stale = conn.execute(
            "DELETE FROM chunk_embeddings WHERE model IS NULL OR model != ?",
            (model_name,),
        ).rowcount
        conn.commit()
        if stale:
            print(f"  Cleared {stale} embedding(s) from other models for re-embed.")

    rows = conn.execute(
        """
        SELECT c.id, c.text_content
        FROM chunks c
        WHERE c.id NOT IN (SELECT chunk_id FROM chunk_embeddings)
        """
    ).fetchall()

    if not rows:
        print("  All chunks already embedded — nothing to do.")
        return 0

    inserted = 0
    cur = conn.cursor()
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        vecs = encode_texts([r["text_content"] for r in batch], model_name)
        for row, vec in zip(batch, vecs):
            cur.execute(
                "INSERT OR IGNORE INTO chunk_embeddings (chunk_id, embedding, model) "
                "VALUES (?, ?, ?)",
                (row["id"], pack_vector(vec.tolist()), model_name),
            )
        conn.commit()
        inserted += len(batch)
        print(f"  Embedded {inserted}/{len(rows)} chunks")

    return inserted
