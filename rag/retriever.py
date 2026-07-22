"""High-level retrieval API for Q&A and summarisation.

Sits between the search primitives (rag.search) and any LLM call:

    Retriever.retrieve()             — ranked, deduplicated chunks for a query
    Retriever.retrieve_for_summary() — all chunks for a document/section,
                                       in reading order
    Retriever.build_context()        — pack chunks into a prompt-ready string
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Literal

from config import (
    CONTEXT_MIN_TRUNCATED_CHARS,
    DEDUP_SIMILARITY_THRESHOLD,
    DEFAULT_ALPHA,
    DEFAULT_CHAR_BUDGET,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    QUERY_EXPANSION_N,
    QUERY_TRANSFORM_ENABLED,
    RERANK_CANDIDATES,
    RERANK_ENABLED,
    RERANK_GUARANTEED_SEMANTIC,
    RERANK_MODEL,
    RRF_K,
)
from rag import db
from rag.search import bm25_search, hybrid_search, semantic_search

SearchMode = Literal["hybrid", "bm25", "semantic"]


@dataclass
class RetrievedChunk:
    """One retrieved chunk plus the metadata needed to cite it.

    Attributes:
        chunk_id (int): Primary key in the chunks table.
        score (float): Relevance score from the search backend.
        text_content (str): Chunk text.
        chunk_type (str): text / heading / ocr_text / table / image_description.
        source (str): Extraction source (text / ocr / vision / ...).
        page_number (int | None): 1-based page number, if known.
        filename (str | None): Source document filename.
        section_title (str | None): Section the chunk belongs to, if detected.
        metadata (dict): Full decoded metadata_json payload.
    """

    chunk_id: int
    score: float
    text_content: str
    chunk_type: str
    source: str
    page_number: int | None
    filename: str | None
    section_title: str | None
    metadata: dict = field(default_factory=dict)

    def citation(self) -> str:
        """Format a human-readable citation for this chunk.

        Returns:
            str: e.g. 'report.pdf · p.4 · "Methods"', or "unknown source".
        """
        parts = []
        if self.filename:
            parts.append(self.filename)
        if self.page_number is not None:
            parts.append(f"p.{self.page_number}")
        if self.section_title:
            parts.append(f'"{self.section_title}"')
        return " · ".join(parts) if parts else "unknown source"


def _deduplicate(chunks: list[RetrievedChunk],
                 sim_threshold: float = DEDUP_SIMILARITY_THRESHOLD) -> list[RetrievedChunk]:
    """Remove near-duplicate chunks by actual text overlap.

    Two chunks are duplicates if one's text is fully contained in the other
    (the inter-block prefix-overlap case), or if they share at least
    ``sim_threshold`` of their tokens (Jaccard similarity). The earlier
    (higher-scored) chunk is kept.

    Note: this compares *content*, not length. An earlier version flagged any
    two chunks of similar length as duplicates, which silently dropped
    distinct paragraphs that happened to be the same size.

    Args:
        chunks (list[RetrievedChunk]): Chunks in score order, best first.
        sim_threshold (float): Token-overlap (Jaccard) threshold for
            near-duplicates.

    Returns:
        list[RetrievedChunk]: Deduplicated chunks, original order preserved.
    """
    kept: list[RetrievedChunk] = []
    kept_tokens: list[set[str]] = []
    for chunk in chunks:
        text = chunk.text_content.strip()
        tokens = set(text.lower().split())
        is_dup = False
        for existing, existing_tokens in zip(kept, kept_tokens):
            existing_text = existing.text_content.strip()
            shorter, longer = sorted([text, existing_text], key=len)
            if shorter and shorter in longer:  # full containment (prefix overlap)
                is_dup = True
                break
            if tokens and existing_tokens:
                overlap = len(tokens & existing_tokens) / len(tokens | existing_tokens)
                if overlap >= sim_threshold:
                    is_dup = True
                    break
        if not is_dup:
            kept.append(chunk)
            kept_tokens.append(tokens)
    return kept


def _to_retrieved(raw: dict) -> RetrievedChunk:
    """Convert a raw search-result dict into a RetrievedChunk.

    Args:
        raw (dict): Result dict from rag.search.

    Returns:
        RetrievedChunk: Typed chunk with decoded metadata.
    """
    meta = json.loads(raw.get("metadata_json") or "{}")
    return RetrievedChunk(
        chunk_id=raw["chunk_id"],
        score=raw["score"],
        text_content=raw["text_content"],
        chunk_type=raw.get("chunk_type", "text"),
        source=raw.get("source", ""),
        page_number=raw.get("page_number"),
        filename=raw.get("filename"),
        section_title=meta.get("section_title"),
        metadata=meta,
    )


class Retriever:
    """Stateful retriever holding an open DB connection.

    The embedding model is loaded lazily on the first semantic or hybrid
    call. Usable as a context manager.
    """

    def __init__(
        self,
        db_path: str,
        model_name: str = EMBEDDING_MODEL,
        default_mode: SearchMode = "hybrid",
        default_alpha: float = DEFAULT_ALPHA,
        rerank: bool = RERANK_ENABLED,
        rerank_model: str = RERANK_MODEL,
        query_transform: bool = QUERY_TRANSFORM_ENABLED,
        same_thread: bool = True,
    ) -> None:
        """Open the database and store retrieval defaults.

        Args:
            db_path (str): Path to the SQLite database.
            model_name (str): Embedding model for semantic search.
            default_mode (SearchMode): Mode used when retrieve() gets none.
            default_alpha (float): Hybrid blend used when retrieve() gets none.
            rerank (bool): Whether retrieve() reranks candidates by default.
            rerank_model (str): Cross-encoder model for reranking.
            query_transform (bool): Whether retrieve() expands the query into
                LLM-generated paraphrases by default.
            same_thread (bool): Pass False to allow the DB connection to be
                used across threads (needed by the Streamlit app).

        Raises:
            FileNotFoundError: If the database file does not exist.
        """
        self.conn = db.connect(db_path, must_exist=True, same_thread=same_thread)
        self.model_name = model_name
        self.default_mode = default_mode
        self.default_alpha = default_alpha
        self.rerank_default = rerank
        self.rerank_model = rerank_model
        self.query_transform_default = query_transform

    def close(self) -> None:
        """Close the underlying database connection.

        Returns:
            None
        """
        self.conn.close()

    def __enter__(self) -> "Retriever":
        """Enter the context manager.

        Returns:
            Retriever: self.
        """
        return self

    def __exit__(self, *_) -> None:
        """Exit the context manager, closing the connection.

        Returns:
            None
        """
        self.close()

    # ------------------------------------------------------------------
    # Q&A retrieval
    # ------------------------------------------------------------------

    def _first_stage_search(
        self,
        query: str,
        candidate_k: int,
        mode: SearchMode,
        alpha: float,
        document_id: int | None,
        guarantee_semantic: bool = False,
    ) -> list[dict]:
        """Run one query through the selected first-stage search backend.

        Args:
            query (str): The (possibly transformed) query phrasing.
            candidate_k (int): Number of candidates to fetch.
            mode (SearchMode): bm25 / semantic / hybrid.
            alpha (float): Hybrid blend weight.
            document_id (int | None): Restrict to one document.
            guarantee_semantic (bool): In hybrid mode, force top semantic hits
                into the pool so a downstream reranker can score them (avoids
                BM25 misses evicting a strong semantic hit during RRF fusion).

        Returns:
            list[dict]: Raw result dicts from the search backend, best-first.
        """
        if mode == "bm25":
            return bm25_search(self.conn, query, top_k=candidate_k,
                               document_id=document_id)
        if mode == "semantic":
            return semantic_search(self.conn, query, top_k=candidate_k,
                                   model_name=self.model_name,
                                   document_id=document_id)
        return hybrid_search(
            self.conn, query, top_k=candidate_k,
            model_name=self.model_name, alpha=alpha, document_id=document_id,
            guaranteed_semantic=RERANK_GUARANTEED_SEMANTIC if guarantee_semantic else 0,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        mode: SearchMode | None = None,
        alpha: float | None = None,
        deduplicate: bool = True,
        document_id: int | None = None,
        rerank: bool | None = None,
        query_transform: bool | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks for a query.

        Pipeline: (optional) expand the query into paraphrases, run first-stage
        search for each and fuse the pools via RRF, deduplicate, then (optional)
        rescore with a cross-encoder before truncating to *top_k*. First-stage
        search casts a wide net; expansion widens it across phrasings; the
        cross-encoder picks the final order.

        When reranking is on, RERANK_CANDIDATES candidates are fetched per
        query; otherwise twice *top_k*, enough to survive deduplication.

        Args:
            query (str): Natural-language query.
            top_k (int): Maximum chunks to return.
            mode (SearchMode | None): bm25 / semantic / hybrid; defaults
                to the instance default.
            alpha (float | None): Hybrid blend weight; defaults to the
                instance default.
            deduplicate (bool): Drop near-duplicate chunks.
            document_id (int | None): Restrict retrieval to one document.
            rerank (bool | None): Rerank candidates with the cross-encoder;
                defaults to the instance default.
            query_transform (bool | None): Expand the query into LLM-generated
                paraphrases; defaults to the instance default.

        Returns:
            list[RetrievedChunk]: Up to top_k chunks, best first.
        """
        mode = mode or self.default_mode
        alpha = alpha if alpha is not None else self.default_alpha
        rerank = self.rerank_default if rerank is None else rerank
        query_transform = (self.query_transform_default
                           if query_transform is None else query_transform)

        # Fetch wide when reranking so the cross-encoder has a deep pool to
        # reorder; otherwise just enough to survive deduplication.
        candidate_k = RERANK_CANDIDATES if rerank else top_k * 2

        # Stage 0: expand the query into paraphrases (original always first).
        if query_transform:
            from rag.query_transform import expand_query
            queries = expand_query(query, n=QUERY_EXPANSION_N)
        else:
            queries = [query]

        # Stage 1: first-stage search per variant. Fuse pools via RRF when
        # there is more than one phrasing; otherwise use the single ranking.
        if len(queries) == 1:
            raw = self._first_stage_search(queries[0], candidate_k, mode,
                                           alpha, document_id,
                                           guarantee_semantic=rerank)
        else:
            from rag.query_transform import fuse_rankings
            ranked_lists = [
                self._first_stage_search(q, candidate_k, mode, alpha,
                                         document_id, guarantee_semantic=rerank)
                for q in queries
            ]
            raw = fuse_rankings(ranked_lists, rrf_k=RRF_K)[:candidate_k]

        chunks = [_to_retrieved(r) for r in raw]
        if deduplicate:
            chunks = _deduplicate(chunks)

        # Stage 2: rerank. Always score against the user's ORIGINAL query, not
        # a paraphrase, so the final ordering reflects what was actually asked.
        if rerank:
            from rag.reranker import rerank as rerank_chunks
            chunks = rerank_chunks(
                query, chunks,
                text_getter=lambda c: c.text_content,
                top_k=top_k, model_name=self.rerank_model,
            )
            return chunks

        return chunks[:top_k]

    # ------------------------------------------------------------------
    # Summarisation retrieval
    # ------------------------------------------------------------------

    def retrieve_for_summary(
        self,
        document_id: int,
        section: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return all chunks for a document in reading order.

        No ranking, full coverage — intended for summarisation.

        Args:
            document_id (int): Document to fetch.
            section (str | None): Restrict to one section title.

        Returns:
            list[RetrievedChunk]: Chunks ordered by page then chunk index,
                each with score 0.0.
        """
        section_filter = (
            "AND json_extract(c.metadata_json, '$.section_title') = ?"
            if section else ""
        )
        params: tuple = (document_id, section) if section else (document_id,)
        cur = self.conn.execute(
            f"""
            SELECT
                c.id AS chunk_id,
                0.0  AS score,
                c.text_content, c.chunk_type, c.source, c.metadata_json,
                p.page_number, d.filename
            FROM chunks c
            LEFT JOIN pages p ON p.id = c.page_id
            LEFT JOIN documents d ON d.id = c.document_id
            WHERE c.document_id = ?
              {section_filter}
            ORDER BY COALESCE(p.page_number, 0), c.chunk_index
            """,
            params,
        )
        return [_to_retrieved(dict(r)) for r in cur]

    # ------------------------------------------------------------------
    # Catalogue helpers
    # ------------------------------------------------------------------

    def list_sections(self, document_id: int) -> list[str]:
        """Return all distinct section titles for a document, in order.

        Args:
            document_id (int): Document to inspect.

        Returns:
            list[str]: Section titles in first-appearance order.
        """
        cur = self.conn.execute(
            """
            SELECT DISTINCT json_extract(c.metadata_json, '$.section_title') AS sec
            FROM chunks c
            WHERE c.document_id = ?
              AND sec IS NOT NULL
            ORDER BY c.id
            """,
            (document_id,),
        )
        return [r["sec"] for r in cur]

    def list_documents(self) -> list[dict]:
        """Return id, filename, and timestamp for every document.

        Returns:
            list[dict]: One dict per document, ordered by id.
        """
        return db.list_documents(self.conn)

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    def build_context(
        self,
        chunks: list[RetrievedChunk],
        char_budget: int = DEFAULT_CHAR_BUDGET,
        include_citations: bool = True,
    ) -> str:
        """Pack chunks into a single prompt-ready string within a budget.

        Each chunk is rendered as a "[citation]" header line followed by
        its text. Chunks are added in the given (score) order until the
        budget is exhausted; the final chunk may be truncated if enough
        room remains.

        Args:
            chunks (list[RetrievedChunk]): Chunks in priority order.
            char_budget (int): Maximum characters of assembled context.
            include_citations (bool): Prepend citation headers.

        Returns:
            str: Assembled context string.
        """
        parts: list[str] = []
        used = 0

        for chunk in chunks:
            header = f"[{chunk.citation()}]" if include_citations else ""
            block = f"{header}\n{chunk.text_content}".strip()
            block_len = len(block) + 2  # +2 for the \n\n separator

            if used + block_len > char_budget:
                remaining = char_budget - used - len(header) - 4
                if remaining >= CONTEXT_MIN_TRUNCATED_CHARS:
                    truncated = f"{header}\n{chunk.text_content[:remaining]}…".strip()
                    parts.append(truncated)
                break

            parts.append(block)
            used += block_len

        return "\n\n".join(parts)
