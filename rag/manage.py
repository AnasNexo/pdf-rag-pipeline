"""High-level corpus management for the UI: add PDFs, list, and reset.

Wraps the existing pipeline stages (ingest -> chunk -> index) behind two
operations a web UI can call directly:

    add_pdfs(paths, db_path)  — ingest, chunk, and index one or more PDFs;
    reset_corpus(db_path)     — delete every document and all derived data.

Each call opens its own short-lived write connection and closes it, so it does
not interfere with a long-lived read connection held elsewhere (e.g. the
Streamlit-cached Retriever, which the caller should rebuild afterwards).
"""

from __future__ import annotations

from dataclasses import dataclass

from rag import db
from rag.chunker import ChunkConfig, chunk_database
from rag.indexer import build_embeddings, build_fts_index


@dataclass
class IngestResult:
    """Outcome of adding PDFs to the corpus.

    Attributes:
        ingested (list[str]): Filenames newly ingested.
        skipped (list[str]): Filenames skipped because identical content was
            already in the database (deduplicated by content hash).
        failed (list[tuple[str, str]]): (filename, error message) per failure.
        chunks_added (int): New chunks built across all ingested PDFs.
        embedded (int): New chunk embeddings created.
    """

    ingested: list[str]
    skipped: list[str]
    failed: list[tuple[str, str]]
    chunks_added: int
    embedded: int


def add_pdfs(paths: list[str], db_path: str,
             chunk_config: ChunkConfig | None = None) -> IngestResult:
    """Ingest, chunk, and index one or more PDFs into the database.

    Mirrors the CLI flow `ingest -> chunk -> index`. Ingestion is per-file so
    one bad PDF does not abort the batch; chunking and indexing then run once
    over whatever was ingested (both are incremental — they only process rows
    that are not yet chunked/embedded).

    Args:
        paths (list[str]): Filesystem paths to PDF files.
        db_path (str): SQLite database path.
        chunk_config (ChunkConfig | None): Chunking parameters (e.g. semantic
            vs word-count). Defaults to ChunkConfig() so existing callers keep
            the config-driven default.

    Returns:
        IngestResult: Per-file outcomes and chunk/embedding counts.
    """
    from rag.pipeline import process_pdf  # heavy imports; defer to call time

    conn = db.connect(db_path)
    db.init_schema(conn)

    ingested: list[str] = []
    skipped: list[str] = []
    failed: list[tuple[str, str]] = []
    try:
        for path in paths:
            name = path.replace("\\", "/").rsplit("/", 1)[-1]
            # process_pdf silently no-ops on a content-hash duplicate, so
            # detect a skip by whether the document count changed.
            before = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            try:
                process_pdf(path, conn)
            except Exception as e:  # noqa: BLE001 — record and keep going
                failed.append((name, str(e)))
                continue
            after = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            (ingested if after > before else skipped).append(name)

        chunks_added = 0
        embedded = 0
        if ingested:
            chunks_added = chunk_database(conn, chunk_config or ChunkConfig())
            build_fts_index(conn)
            embedded = build_embeddings(conn)
    finally:
        conn.close()

    return IngestResult(ingested=ingested, skipped=skipped, failed=failed,
                        chunks_added=chunks_added, embedded=embedded)


def list_documents(db_path: str) -> list[dict]:
    """Return all documents currently in the database.

    Args:
        db_path (str): SQLite database path.

    Returns:
        list[dict]: One dict per document (id, filename, processed_at);
            empty if the database does not exist yet.
    """
    import os
    if not os.path.isfile(db_path):
        return []
    conn = db.connect(db_path)
    try:
        # The file can exist but be empty (e.g. a fresh container where the DB
        # was created but never ingested into) — ensure the schema exists so
        # the query below doesn't fail with "no such table: documents".
        db.init_schema(conn)
        return db.list_documents(conn)
    finally:
        conn.close()


def rechunk_corpus(db_path: str, semantic: bool) -> tuple[int, int]:
    """Rebuild chunks (and their indexes) in place with a new splitter method.

    Switching chunking method does NOT require re-parsing the PDFs — the parsed
    semantic_blocks already live in the database. This re-runs only the fast
    tail of the pipeline: reset chunks (clearing the FTS + embedding rows they
    own), re-chunk from the existing blocks with the chosen method, then rebuild
    both indexes. Far cheaper than delete + re-upload, which re-does OCR/vision.

    Args:
        db_path (str): SQLite database path.
        semantic (bool): True for topic-aware chunking, False for word-count.

    Returns:
        tuple[int, int]: (chunks_built, chunks_embedded).
    """
    import os
    if not os.path.isfile(db_path):
        return 0, 0
    conn = db.connect(db_path)
    try:
        chunks_built = chunk_database(
            conn, ChunkConfig(reset=True, semantic=semantic))
        build_fts_index(conn)
        embedded = build_embeddings(conn, reembed=True)
        return chunks_built, embedded
    finally:
        conn.close()


def reset_corpus(db_path: str) -> int:
    """Delete every document and all derived data from the database.

    Removes documents one by one via the existing cascade delete, which also
    clears the FTS index and embeddings — leaving an empty but valid schema
    rather than an orphaned database file.

    Args:
        db_path (str): SQLite database path.

    Returns:
        int: Number of documents deleted.
    """
    import os
    if not os.path.isfile(db_path):
        return 0
    conn = db.connect(db_path)
    try:
        docs = db.list_documents(conn)
        for d in docs:
            db.delete_document(conn, d["id"])
        return len(docs)
    finally:
        conn.close()
