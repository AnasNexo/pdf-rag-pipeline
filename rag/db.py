"""SQLite storage layer: connections, schema, migrations, deletion, diagnostics.

All table definitions for the pipeline live here so the full schema can be
read in one place:

    documents        — one row per ingested PDF (deduplicated by content hash)
    pages            — extracted/OCR'd text per page
    images           — image descriptions (OCR or vision model)
    tables           — extracted tables (markdown + raw cells)
    semantic_blocks  — typed text blocks (text / heading / table / image / OCR)
    chunks           — retrieval chunks built from semantic blocks
    chunk_embeddings — packed float vectors per chunk
    chunks_fts       — FTS5 virtual table for BM25 search
"""

from __future__ import annotations

import os
import sqlite3


def connect(db_path: str, must_exist: bool = False,
            same_thread: bool = True) -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys and WAL mode enabled.

    Args:
        db_path (str): Path to the SQLite database file.
        must_exist (bool): If True, raise instead of creating a new file.
        same_thread (bool): If False, allow the connection to be used from
            threads other than the one that created it. Needed by the
            Streamlit app, where a cached connection is reused across the
            different threads Streamlit runs reruns on. Safe here because
            access is read-only and serialised; do not set False for
            concurrent writers.

    Returns:
        sqlite3.Connection: Connection with row_factory set to sqlite3.Row.

    Raises:
        FileNotFoundError: If must_exist is True and the file is missing.
    """
    if must_exist and not os.path.isfile(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
    conn = sqlite3.connect(db_path, check_same_thread=same_thread)
    conn.row_factory = sqlite3.Row
    # Foreign key enforcement is OFF by default in SQLite; without it,
    # child rows with orphaned parent ids are silently accepted.
    conn.execute("PRAGMA foreign_keys = ON")
    # WAL mode avoids partial-write corruption on crashes.
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create all ingestion tables and run in-place column migrations.

    Safe to call repeatedly: uses CREATE TABLE IF NOT EXISTS and only adds
    columns that are missing.

    Args:
        conn (sqlite3.Connection): Open database connection.

    Returns:
        None
    """
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            content_hash TEXT,
            page_count INTEGER,
            processed_at TEXT
        )
    """)
    _ensure_column(cur, "documents", "content_hash", "TEXT")
    cur.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_content_hash "
        "ON documents(content_hash)"
    )

    cur.execute("""
        CREATE TABLE IF NOT EXISTS pages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            page_number INTEGER NOT NULL,
            source TEXT NOT NULL,
            text_content TEXT,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            page_id INTEGER NOT NULL,
            image_index INTEGER NOT NULL,
            bbox TEXT,
            width INTEGER,
            height INTEGER,
            description TEXT,
            description_source TEXT,
            FOREIGN KEY (page_id) REFERENCES pages(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS tables (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            page_id INTEGER NOT NULL,
            table_index INTEGER NOT NULL,
            bbox TEXT,
            row_count INTEGER,
            col_count INTEGER,
            header_names TEXT,
            markdown TEXT NOT NULL,
            cells_json TEXT,
            source TEXT NOT NULL,
            accuracy REAL,
            FOREIGN KEY (document_id) REFERENCES documents(id),
            FOREIGN KEY (page_id) REFERENCES pages(id)
        )
    """)
    _ensure_column(cur, "tables", "document_id", "INTEGER REFERENCES documents(id)")
    _ensure_column(cur, "tables", "accuracy", "REAL")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_tables_page ON tables(page_id, table_index)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_tables_document ON tables(document_id)"
    )

    cur.execute("""
        CREATE TABLE IF NOT EXISTS semantic_blocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            page_id INTEGER,
            image_id INTEGER,
            table_id INTEGER,
            block_index INTEGER NOT NULL,
            block_type TEXT NOT NULL,
            source TEXT NOT NULL,
            bbox TEXT,
            text_content TEXT NOT NULL,
            confidence REAL,
            section_title TEXT,
            created_at TEXT,
            FOREIGN KEY (document_id) REFERENCES documents(id),
            FOREIGN KEY (page_id) REFERENCES pages(id),
            FOREIGN KEY (image_id) REFERENCES images(id),
            FOREIGN KEY (table_id) REFERENCES tables(id)
        )
    """)
    _ensure_column(cur, "semantic_blocks", "section_title", "TEXT")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_semantic_blocks_document_page "
        "ON semantic_blocks(document_id, page_id, block_index)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_semantic_blocks_type "
        "ON semantic_blocks(block_type)"
    )

    conn.commit()


def ensure_chunk_table(conn: sqlite3.Connection) -> None:
    """Create the chunks table and its indexes if they do not exist.

    Args:
        conn (sqlite3.Connection): Open database connection.

    Returns:
        None
    """
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            page_id INTEGER,
            semantic_block_id INTEGER,
            table_id INTEGER,
            image_id INTEGER,
            chunk_index INTEGER NOT NULL,
            chunk_type TEXT NOT NULL,
            source TEXT NOT NULL,
            start_word INTEGER,
            end_word INTEGER,
            text_content TEXT NOT NULL,
            metadata_json TEXT,
            created_at TEXT,
            FOREIGN KEY (document_id) REFERENCES documents(id),
            FOREIGN KEY (page_id) REFERENCES pages(id),
            FOREIGN KEY (semantic_block_id) REFERENCES semantic_blocks(id),
            FOREIGN KEY (table_id) REFERENCES tables(id),
            FOREIGN KEY (image_id) REFERENCES images(id)
        )
    """)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_document_page "
        "ON chunks(document_id, page_id, chunk_index)"
    )
    cur.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_block_unique "
        "ON chunks(semantic_block_id, chunk_index)"
    )
    conn.commit()


def ensure_embedding_table(conn: sqlite3.Connection) -> None:
    """Create the chunk_embeddings table if it does not exist.

    Args:
        conn (sqlite3.Connection): Open database connection.

    Returns:
        None
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunk_embeddings (
            chunk_id INTEGER PRIMARY KEY,
            embedding BLOB NOT NULL,
            model TEXT NOT NULL,
            FOREIGN KEY (chunk_id) REFERENCES chunks(id)
        )
    """)
    conn.commit()


def ensure_fts_table(conn: sqlite3.Connection) -> None:
    """Create the FTS5 virtual table for BM25 search if it does not exist.

    Porter stemming is enabled so 'running' matches 'run', etc. The FTS
    rowid mirrors chunks.id so the two tables join directly.

    Args:
        conn (sqlite3.Connection): Open database connection.

    Returns:
        None
    """
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text_content,
            tokenize='porter unicode61'
        )
    """)
    conn.commit()


def _ensure_column(cur: sqlite3.Cursor, table: str, column: str, ddl_type: str) -> None:
    """Add a column to a table if it is missing (lightweight migration).

    Args:
        cur (sqlite3.Cursor): Active cursor.
        table (str): Table name.
        column (str): Column name to ensure.
        ddl_type (str): Column type/constraint DDL fragment.

    Returns:
        None
    """
    cur.execute(f"PRAGMA table_info({table})")
    existing = [row[1] for row in cur.fetchall()]
    if column not in existing:
        print(f"[migration] Adding '{column}' column to '{table}' table...")
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}")


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    """Check whether a table exists in the database.

    Args:
        conn (sqlite3.Connection): Open database connection.
        name (str): Table name to look up.

    Returns:
        bool: True if the table exists.
    """
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def list_documents(conn: sqlite3.Connection) -> list[dict]:
    """Return id, filename, and processing timestamp for every document.

    Args:
        conn (sqlite3.Connection): Open database connection.

    Returns:
        list[dict]: One dict per document, ordered by id.
    """
    cur = conn.execute(
        "SELECT id, filename, processed_at FROM documents ORDER BY id"
    )
    return [dict(r) for r in cur]


def find_document(conn: sqlite3.Connection, ref: str) -> dict | None:
    """Look up a document by numeric id or by filename.

    Args:
        conn (sqlite3.Connection): Open database connection.
        ref (str): Document id (digits) or exact filename
            (case-insensitive).

    Returns:
        dict | None: The matching document row as a dict, or None.
    """
    ref = ref.strip()
    if ref.lstrip("-").isdigit():
        row = conn.execute(
            "SELECT id, filename FROM documents WHERE id = ?", (int(ref),)
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT id, filename FROM documents WHERE LOWER(filename) = LOWER(?)",
            (ref,),
        ).fetchone()
    return dict(row) if row else None


def delete_document(conn: sqlite3.Connection, document_id: int) -> None:
    """Remove a document and every derived row across all tables.

    Cleans the FTS5 index and the embedding table as well, so the document
    can be reprocessed from scratch (used by --force) or permanently removed.

    Args:
        conn (sqlite3.Connection): Open database connection.
        document_id (int): Primary key of the document to delete.

    Returns:
        None
    """
    cur = conn.cursor()

    # Collect chunk ids first — needed to clean FTS5 and embeddings, which
    # have no document_id column of their own.
    if _table_exists(conn, "chunks"):
        chunk_ids = [r[0] for r in cur.execute(
            "SELECT id FROM chunks WHERE document_id = ?", (document_id,)
        )]
        if chunk_ids:
            placeholders = ",".join("?" * len(chunk_ids))
            if _table_exists(conn, "chunks_fts"):
                cur.execute(
                    f"DELETE FROM chunks_fts WHERE rowid IN ({placeholders})",
                    chunk_ids,
                )
            if _table_exists(conn, "chunk_embeddings"):
                cur.execute(
                    f"DELETE FROM chunk_embeddings WHERE chunk_id IN ({placeholders})",
                    chunk_ids,
                )
        cur.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))

    # Delete child tables in FK order.
    cur.execute("DELETE FROM semantic_blocks WHERE document_id = ?", (document_id,))
    cur.execute(
        "DELETE FROM tables WHERE page_id IN "
        "(SELECT id FROM pages WHERE document_id = ?)", (document_id,)
    )
    cur.execute(
        "DELETE FROM images WHERE page_id IN "
        "(SELECT id FROM pages WHERE document_id = ?)", (document_id,)
    )
    cur.execute("DELETE FROM pages WHERE document_id = ?", (document_id,))
    cur.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    conn.commit()


def diagnose(db_path: str) -> None:
    """Print a summary of database contents and check for orphaned rows.

    Args:
        db_path (str): Path to the SQLite database file.

    Returns:
        None
    """
    if not os.path.isfile(db_path):
        print(f"[diagnose] Database not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    print(f"\n=== Diagnosing: {db_path} ===\n")

    cur.execute("SELECT id, filename, page_count, processed_at FROM documents")
    docs = cur.fetchall()
    print(f"Documents ({len(docs)} total):")
    for doc_id, fname, page_count, ts in docs:
        cur.execute("SELECT COUNT(*) FROM pages WHERE document_id = ?", (doc_id,))
        pg_count = cur.fetchone()[0]
        cur.execute(
            "SELECT COUNT(*) FROM images WHERE page_id IN "
            "(SELECT id FROM pages WHERE document_id = ?)", (doc_id,)
        )
        img_count = cur.fetchone()[0]
        print(f"  id={doc_id}  file='{fname}'  "
              f"expected_pages={page_count}  stored_pages={pg_count}  "
              f"images={img_count}  processed={ts}")

    print()
    for table in ("pages", "images", "tables", "semantic_blocks"):
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        print(f"Total {table} rows: {cur.fetchone()[0]}")

    cur.execute(
        "SELECT COUNT(*) FROM pages WHERE document_id NOT IN (SELECT id FROM documents)"
    )
    orphan_pages = cur.fetchone()[0]
    cur.execute(
        "SELECT COUNT(*) FROM images WHERE page_id NOT IN (SELECT id FROM pages)"
    )
    orphan_images = cur.fetchone()[0]
    if orphan_pages or orphan_images:
        print(f"\n[WARNING] Orphaned rows — orphan pages: {orphan_pages}, "
              f"orphan images: {orphan_images}")
        print("  These rows exist but are unreachable via their parent FK.")
        print("  This is the classic symptom of lastrowid=0 or FK enforcement OFF.")
    else:
        print("\nNo orphaned rows found.")

    conn.close()
