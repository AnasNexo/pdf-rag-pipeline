"""Build retrieval-friendly chunks from semantic blocks.

Consumes the `semantic_blocks` table written by the ingestion pipeline and
writes a `chunks` table. Key properties:

- Text blocks are split at sentence boundaries (not raw word windows),
  with word-count overlap carried across sentence edges.
- The last `overlap_words` words of each non-atomic block are prepended as
  context to the first chunk of the next block on the same page and
  section (inter-block overlap).
- Table and image-description blocks are kept atomic (never split).
- `section_title` detected during parsing is forwarded into each chunk's
  `metadata_json` so downstream Q&A can cite the originating section.
- Re-running without reset is safe: duplicate (semantic_block_id,
  chunk_index) pairs are silently ignored via INSERT OR IGNORE.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from config import (
    CHUNK_COMMIT_BATCH_SIZE,
    CHUNK_MAX_WORDS,
    CHUNK_OVERLAP_WORDS,
    SEMANTIC_CHUNK_MIN_SIMILARITY,
    SEMANTIC_CHUNK_PERCENTILE,
    SEMANTIC_CHUNKING_ENABLED,
)
from rag.db import ensure_chunk_table

WORD_RE = re.compile(r"\S+")
# Split after sentence-ending punctuation followed by whitespace.
SENTENCE_END_RE = re.compile(r"(?<=[.!?…])\s+")

#: Block types that are never split into multiple chunks.
ATOMIC_BLOCK_TYPES = frozenset({"table", "image_description"})


@dataclass(frozen=True)
class ChunkConfig:
    """Parameters controlling one chunking run.

    Attributes:
        max_words (int): Maximum words per chunk.
        overlap_words (int): Words of overlap between adjacent chunks.
        reset (bool): Delete existing chunks before rebuilding.
        document_id (int | None): Restrict the run to one document.
        semantic (bool): Cut at topic shifts (embed sentences, split on
            similarity drops) instead of purely by word count. max_words still
            caps each chunk as a safety net.
        semantic_percentile (float): Percentile of adjacent-sentence similarity
            drops at which to cut when ``semantic`` is on (document-adaptive).
    """

    max_words: int = CHUNK_MAX_WORDS
    overlap_words: int = CHUNK_OVERLAP_WORDS
    reset: bool = False
    document_id: int | None = None
    semantic: bool = SEMANTIC_CHUNKING_ENABLED
    semantic_percentile: float = SEMANTIC_CHUNK_PERCENTILE


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def word_tokens(text: str) -> list[str]:
    """Tokenise text into whitespace-delimited words.

    Args:
        text (str): Input text (may be None or empty).

    Returns:
        list[str]: Word tokens in order.
    """
    return WORD_RE.findall((text or "").strip())


def sentence_split(text: str) -> list[str]:
    """Split text into sentences on punctuation boundaries.

    Args:
        text (str): Input text (may be None or empty).

    Returns:
        list[str]: Non-empty sentence strings in order.
    """
    parts = SENTENCE_END_RE.split((text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def is_atomic_block(block_type: str) -> bool:
    """Check whether a block type must stay as a single chunk.

    Args:
        block_type (str): Semantic block type.

    Returns:
        bool: True for tables and image descriptions.
    """
    return block_type in ATOMIC_BLOCK_TYPES


# ---------------------------------------------------------------------------
# Sentence-aware chunking
# ---------------------------------------------------------------------------

def split_into_sentence_chunks(
    text: str,
    max_words: int,
    overlap_words: int,
) -> list[tuple[str, int, int]]:
    """Split text into overlapping chunks that end at sentence boundaries.

    Sentences longer than *max_words* are hard-split into sliding word
    windows; otherwise sentences are packed greedily and the last
    *overlap_words* words of each chunk seed the next one.

    Args:
        text (str): Text to split.
        max_words (int): Maximum words per chunk.
        overlap_words (int): Words of overlap between adjacent chunks.

    Returns:
        list[tuple[str, int, int]]: (chunk_text, start_word, end_word)
            tuples, where word indices are 0-based and end_word inclusive.
    """
    sentences = sentence_split(text)
    if not sentences:
        return []

    tokenized = [(s, word_tokens(s)) for s in sentences]
    tokenized = [(s, w) for s, w in tokenized if w]
    if not tokenized:
        return []

    result: list[tuple[str, int, int]] = []
    word_offset = 0      # absolute index of the first word in curr_words
    curr_words: list[str] = []

    for _sent_text, sent_words in tokenized:
        if len(sent_words) > max_words:
            # Flush buffer, then slice the long sentence by word windows.
            if curr_words:
                start = word_offset
                result.append((" ".join(curr_words), start, start + len(curr_words) - 1))
                word_offset += len(curr_words)
                curr_words = []
            step = max(1, max_words - overlap_words)
            for i in range(0, len(sent_words), step):
                window = sent_words[i:i + max_words]
                start = word_offset + i
                result.append((" ".join(window), start, start + len(window) - 1))
            word_offset += len(sent_words)
            continue

        if curr_words and len(curr_words) + len(sent_words) > max_words:
            # Emit what we have, then start the next chunk with the overlap suffix.
            start = word_offset
            result.append((" ".join(curr_words), start, start + len(curr_words) - 1))
            overlap_slice = curr_words[-overlap_words:] if overlap_words else []
            word_offset += len(curr_words) - len(overlap_slice)
            curr_words = overlap_slice + list(sent_words)
        else:
            curr_words.extend(sent_words)

    if curr_words:
        start = word_offset
        result.append((" ".join(curr_words), start, start + len(curr_words) - 1))

    return result


# ---------------------------------------------------------------------------
# Semantic (topic-aware) chunking
# ---------------------------------------------------------------------------

def _semantic_cut_points(
    sentences: list[str],
    percentile: float,
) -> set[int]:
    """Find sentence indices where the topic shifts, via embedding similarity.

    Each sentence is embedded and the cosine similarity between every adjacent
    pair is measured. A *low* similarity (a valley) marks a topic boundary. The
    threshold is document-adaptive: cut where the drop is sharper than the given
    percentile of all drops in this block, falling back to an absolute floor
    when there are too few sentences for a stable percentile.

    Args:
        sentences (list[str]): Sentences in reading order.
        percentile (float): Percentile (0-100) of similarity drops to cut at.

    Returns:
        set[int]: Indices ``i`` meaning "start a new chunk before sentence i".
    """
    if len(sentences) < 3:
        return set()  # too short to detect a meaningful boundary

    # Imported here so the fast word-count path never pulls in numpy / the
    # embedding model unless semantic chunking is actually requested.
    import numpy as np

    from rag.embedder import encode_texts

    vecs = np.asarray(encode_texts(sentences), dtype=np.float32)
    # Vectors are L2-normalised at encode time, so a dot product is cosine.
    sims = np.array([float(vecs[i] @ vecs[i - 1])
                     for i in range(1, len(sentences))])
    drops = 1.0 - sims  # larger drop = sharper topic change

    # Adaptive threshold from this block's own distribution of drops, floored
    # by the absolute minimum so a uniformly on-topic block is not over-cut.
    cutoff = float(np.percentile(drops, percentile))
    floor = 1.0 - SEMANTIC_CHUNK_MIN_SIMILARITY
    cutoff = max(cutoff, floor)

    # drops[i-1] compares sentence i to i-1, so a cut sits *before* sentence i.
    return {i for i in range(1, len(sentences)) if drops[i - 1] >= cutoff}


def semantic_split(
    text: str,
    max_words: int,
    overlap_words: int,
    percentile: float = SEMANTIC_CHUNK_PERCENTILE,
) -> list[tuple[str, int, int]]:
    """Split text at topic boundaries, capped by *max_words*.

    A drop-in alternative to :func:`split_into_sentence_chunks` with the same
    return shape. Sentences are grouped until either a semantic boundary is
    reached or the running chunk would exceed *max_words*; then the chunk is
    emitted and the last *overlap_words* words seed the next one (identical
    overlap behaviour to the word-count splitter, so retrieval stays consistent).

    Args:
        text (str): Text to split.
        max_words (int): Hard cap on words per chunk (safety net).
        overlap_words (int): Words of overlap between adjacent chunks.
        percentile (float): Percentile of similarity drops at which to cut.

    Returns:
        list[tuple[str, int, int]]: (chunk_text, start_word, end_word) tuples,
            word indices 0-based and end_word inclusive.
    """
    sentences = sentence_split(text)
    if not sentences:
        return []

    tokenized = [(s, word_tokens(s)) for s in sentences]
    tokenized = [(s, w) for s, w in tokenized if w]
    if not tokenized:
        return []

    cut_before = _semantic_cut_points([s for s, _ in tokenized], percentile)

    result: list[tuple[str, int, int]] = []
    word_offset = 0
    curr_words: list[str] = []

    def _flush() -> list[str]:
        """Emit the current chunk and return the overlap suffix for the next."""
        nonlocal word_offset
        start = word_offset
        result.append((" ".join(curr_words), start, start + len(curr_words) - 1))
        overlap_slice = curr_words[-overlap_words:] if overlap_words else []
        word_offset += len(curr_words) - len(overlap_slice)
        return overlap_slice

    for idx, (_sent_text, sent_words) in enumerate(tokenized):
        # A single over-long sentence: flush, then hard-window it (same as the
        # word-count splitter, so no sentence is ever lost).
        if len(sent_words) > max_words:
            if curr_words:
                _flush()
                curr_words = []
            step = max(1, max_words - overlap_words)
            for i in range(0, len(sent_words), step):
                window = sent_words[i:i + max_words]
                start = word_offset + i
                result.append((" ".join(window), start, start + len(window) - 1))
            word_offset += len(sent_words)
            continue

        # Cut here if this sentence starts a new topic, or would overflow.
        boundary = idx in cut_before or (
            curr_words and len(curr_words) + len(sent_words) > max_words)
        if boundary and curr_words:
            curr_words = _flush() + list(sent_words)
        else:
            curr_words.extend(sent_words)

    if curr_words:
        start = word_offset
        result.append((" ".join(curr_words), start, start + len(curr_words) - 1))

    return result


# ---------------------------------------------------------------------------
# Block → chunks
# ---------------------------------------------------------------------------

def _safe_section_title(block: sqlite3.Row) -> str | None:
    """Read a block's section title, tolerating pre-migration rows.

    Args:
        block (sqlite3.Row): Semantic block row.

    Returns:
        str | None: Section title, or None if the column is absent.
    """
    try:
        return block["section_title"]
    except (IndexError, KeyError):
        return None


def build_chunks_for_block(
    block: sqlite3.Row,
    config: ChunkConfig,
    prefix_words: list[str] | None = None,
) -> list[dict]:
    """Convert one semantic block into a list of chunk dicts.

    Args:
        block (sqlite3.Row): Semantic block row from the database.
        config (ChunkConfig): Chunking parameters.
        prefix_words (list[str] | None): Last N words of the previous
            non-atomic block on the same page/section, prepended as
            context to the first chunk so cross-block retrieval has
            overlap even at block boundaries.

    Returns:
        list[dict]: Chunk dicts ready for insertion. Atomic blocks yield
            exactly one chunk; empty blocks yield none.
    """
    text = (block["text_content"] or "").strip()
    if not text:
        return []

    block_type = block["block_type"]
    section_title = _safe_section_title(block)

    if is_atomic_block(block_type):
        return [{
            "chunk_index": 0,
            "chunk_type": block_type,
            "source": block["source"],
            "start_word": 0,
            "end_word": max(0, len(word_tokens(text)) - 1),
            "text_content": text,
            "metadata_json": json.dumps({
                "block_type": block_type,
                "atomic": True,
                "section_title": section_title,
            }, ensure_ascii=True),
        }]

    # Prepend inter-block context (kept out of word-index accounting).
    prefix_count = 0
    if prefix_words:
        prefix_count = len(prefix_words)
        full_text = " ".join(prefix_words) + " " + text
    else:
        full_text = text

    if config.semantic:
        windows = semantic_split(full_text, config.max_words,
                                 config.overlap_words,
                                 percentile=config.semantic_percentile)
    else:
        windows = split_into_sentence_chunks(
            full_text, config.max_words, config.overlap_words)
    method = "semantic" if config.semantic else "word_count"

    chunks = []
    for chunk_index, (chunk_text, start_word, end_word) in enumerate(windows):
        if not chunk_text.strip():
            continue
        chunks.append({
            "chunk_index": chunk_index,
            "chunk_type": block_type,
            "source": block["source"],
            # Adjust indices to be relative to the block's own words.
            "start_word": max(0, start_word - prefix_count),
            "end_word": max(0, end_word - prefix_count),
            "text_content": chunk_text,
            "metadata_json": json.dumps({
                "block_type": block_type,
                "chunk_method": method,
                "overlap_words": config.overlap_words,
                "max_words": config.max_words,
                "source_block_id": block["id"],
                "inter_block_prefix_words": prefix_count,
                "section_title": section_title,
            }, ensure_ascii=True),
        })

    return chunks


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def reset_chunks(conn: sqlite3.Connection, document_id: int | None = None) -> None:
    """Delete existing chunks and their derived index rows.

    The FTS index (chunks_fts) and embeddings (chunk_embeddings) reference
    chunk ids, so they must be cleared first or ``DELETE FROM chunks`` fails the
    foreign-key constraint (and would otherwise leave orphaned index rows that
    survive into the next run). Mirrors db.delete_document's cleanup, scoped to
    the chunk layer so re-chunking (e.g. switching splitter method) is clean.

    Args:
        conn (sqlite3.Connection): Open database connection.
        document_id (int | None): Restrict deletion to this document, or
            delete all chunks when None.

    Returns:
        None
    """
    def _table_exists(name: str) -> bool:
        return conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=?",
            (name,),
        ).fetchone() is not None

    # Collect the chunk ids being removed to clean their index rows first.
    if document_id is not None:
        chunk_ids = [r[0] for r in conn.execute(
            "SELECT id FROM chunks WHERE document_id = ?", (document_id,))]
    else:
        chunk_ids = [r[0] for r in conn.execute("SELECT id FROM chunks")]

    if chunk_ids:
        placeholders = ",".join("?" * len(chunk_ids))
        if _table_exists("chunks_fts"):
            conn.execute(
                f"DELETE FROM chunks_fts WHERE rowid IN ({placeholders})",
                chunk_ids)
        if _table_exists("chunk_embeddings"):
            conn.execute(
                f"DELETE FROM chunk_embeddings WHERE chunk_id IN ({placeholders})",
                chunk_ids)

    if document_id is not None:
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
    else:
        conn.execute("DELETE FROM chunks")
    conn.commit()


def insert_chunks(conn: sqlite3.Connection, block: sqlite3.Row, chunks: list[dict]) -> None:
    """Insert chunk dicts derived from one semantic block.

    Uses INSERT OR IGNORE so re-running the chunker never duplicates
    (semantic_block_id, chunk_index) pairs.

    Args:
        conn (sqlite3.Connection): Open database connection.
        block (sqlite3.Row): The originating semantic block row.
        chunks (list[dict]): Chunk dicts from build_chunks_for_block.

    Returns:
        None
    """
    if not chunks:
        return
    cur = conn.cursor()
    for chunk in chunks:
        cur.execute(
            """
            INSERT OR IGNORE INTO chunks
                (document_id, page_id, semantic_block_id, table_id, image_id, chunk_index,
                 chunk_type, source, start_word, end_word, text_content, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                block["document_id"],
                block["page_id"],
                block["id"],
                block["table_id"],
                block["image_id"],
                chunk["chunk_index"],
                chunk["chunk_type"],
                chunk["source"],
                chunk["start_word"],
                chunk["end_word"],
                chunk["text_content"],
                chunk["metadata_json"],
                datetime.now(timezone.utc).isoformat(),
            ),
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def chunk_database(conn: sqlite3.Connection, config: ChunkConfig) -> int:
    """Chunk every semantic block in the database (or one document).

    Walks blocks in reading order, maintaining inter-block overlap that
    resets at page and section boundaries so off-topic context never
    bleeds into the first chunk of a new section.

    Args:
        conn (sqlite3.Connection): Open database connection.
        config (ChunkConfig): Chunking parameters.

    Returns:
        int: Number of chunks produced (before OR IGNORE dedup).
    """
    ensure_chunk_table(conn)
    method = "semantic (topic-aware)" if config.semantic else "word-count"
    print(f"  Chunking method: {method} "
          f"(max_words={config.max_words}, overlap={config.overlap_words})")
    if config.reset:
        reset_chunks(conn, config.document_id)

    cur = conn.cursor()
    base_query = """
        SELECT sb.id, sb.document_id, sb.page_id, sb.image_id, sb.table_id,
               sb.block_index, sb.block_type, sb.source, sb.bbox,
               sb.text_content, sb.section_title
        FROM semantic_blocks sb
        {where}
        ORDER BY sb.document_id, COALESCE(sb.page_id, 0), sb.block_index, sb.id
    """
    if config.document_id is not None:
        cur.execute(base_query.format(where="WHERE sb.document_id = ?"),
                    (config.document_id,))
    else:
        cur.execute(base_query.format(where=""))

    inserted = 0
    batch = 0
    prev_page_id: int | None = None
    prev_section: str | None = None
    tail_words: list[str] = []

    for block in cur:
        current_section = _safe_section_title(block)

        # Reset inter-block context at page or section boundaries.
        if block["page_id"] != prev_page_id or current_section != prev_section:
            tail_words = []

        prefix = tail_words if not is_atomic_block(block["block_type"]) else []
        chunks = build_chunks_for_block(block, config, prefix_words=prefix)
        insert_chunks(conn, block, chunks)

        # Update the tail for the next block — atomic blocks (tables/images)
        # never feed overlap into the following text chunk.
        if chunks and not is_atomic_block(block["block_type"]):
            tail_words = word_tokens(chunks[-1]["text_content"])[-config.overlap_words:]
        else:
            tail_words = []

        prev_page_id = block["page_id"]
        prev_section = current_section
        inserted += len(chunks)
        batch += len(chunks)
        if batch >= CHUNK_COMMIT_BATCH_SIZE:
            conn.commit()
            batch = 0

    conn.commit()
    return inserted
