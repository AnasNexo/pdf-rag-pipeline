"""Central configuration for the RAG pipeline.

Every tunable constant lives here: file paths, page-classification
thresholds, chunking sizes, embedding model names, retrieval parameters,
and LLM settings. Modules import from this file instead of hardcoding
values, so behaviour can be changed in one place.
"""

# ---------------------------------------------------------------------------
# Paths & environment
# ---------------------------------------------------------------------------

import os as _os

#: Default SQLite database file. Overridable via the RAG_DB_PATH environment
#: variable so a container can point it at a persisted volume (see
#: docker-compose.yml); defaults to a file in the working directory.
DEFAULT_DB_PATH = _os.environ.get("RAG_DB_PATH", "pdf_data.db")

#: Directory holding raw PDF documents.
DATA_DIR = "data"

# LLM API keys are NOT configured here — they are read from the environment by
# rag.llm, following the PROVIDER_CHAIN / VISION_PROVIDER_CHAIN lists defined
# lower in this file. See ".env.example" for the variable names.

# ---------------------------------------------------------------------------
# Page classification (text layer vs. OCR)
# ---------------------------------------------------------------------------

#: Minimum word count for a PDF text layer to be considered "dominant"
#: (i.e. trustworthy enough to skip OCR).
TEXT_LAYER_WORD_THRESHOLD = 30

#: Minimum character count for a PDF text layer to be considered dominant.
TEXT_LAYER_CHAR_THRESHOLD = 180

#: Minimum OCR word count for an image to be classified as text-dominant.
OCR_TEXT_WORD_THRESHOLD = 20

#: Minimum fraction of image area covered by OCR text boxes for an image
#: to be classified as text-dominant.
OCR_TEXT_AREA_THRESHOLD = 0.25

#: Number of significant vector drawings that flags a page as visual.
VECTOR_DRAWING_COUNT_THRESHOLD = 4

#: Fraction of page area covered by vector drawings that flags a page as visual.
VECTOR_DRAWING_COVERAGE_THRESHOLD = 0.03

#: Minimum (drawing area / page area) ratio for a single vector drawing
#: to count as "significant".
SIGNIFICANT_DRAWING_AREA_RATIO = 0.004

#: DPI used when rasterising a page for OCR or vision description.
PAGE_RENDER_DPI = 200

#: Embedded images smaller than this (either dimension, px) are skipped.
MIN_IMAGE_DIMENSION_PX = 50

# ---------------------------------------------------------------------------
# Semantic block extraction
# ---------------------------------------------------------------------------

#: Maximum characters per semantic block when splitting OCR text.
SEMANTIC_BLOCK_MAX_CHARS = 1200

#: A block is a heading if its average font size exceeds the page median
#: by this ratio (and it is short enough).
HEADING_SIZE_RATIO = 1.2

#: Maximum word count for a large-font block to qualify as a heading.
HEADING_MAX_WORDS = 20

#: Maximum word count for a bold block to qualify as a heading.
BOLD_HEADING_MAX_WORDS = 15

#: Minimum (avg size / median size) ratio for a bold block to qualify
#: as a heading.
BOLD_HEADING_MIN_SIZE_RATIO = 0.95

# ---------------------------------------------------------------------------
# Table extraction
# ---------------------------------------------------------------------------

#: Camelot "stream" tables below this accuracy score are discarded
#: (Camelot's own confidence metric, 0-100).
CAMELOT_STREAM_MIN_ACCURACY = 85.0

#: Camelot stream-flavor tolerance settings.
CAMELOT_STREAM_EDGE_TOL = 50
CAMELOT_STREAM_ROW_TOL = 10

#: pdfplumber snap/join tolerances for grid-based table detection.
PDFPLUMBER_SNAP_TOLERANCE = 3
PDFPLUMBER_JOIN_TOLERANCE = 3

#: Grid validation: reject "tables" whose average non-empty cell exceeds
#: this many characters (paragraph text masquerading as a table).
TABLE_MAX_AVG_CELL_CHARS = 200

#: Grid validation: reject tables containing any single cell longer than this.
TABLE_MAX_CELL_CHARS = 500

#: Grid validation: maximum allowed fraction of empty cells.
TABLE_MAX_EMPTY_CELL_RATIO = 0.6

#: Grid validation: maximum allowed fraction of rows with inconsistent
#: column counts.
TABLE_MAX_INCONSISTENT_ROW_RATIO = 0.3

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

#: Maximum words per retrieval chunk.
CHUNK_MAX_WORDS = 300

#: Words of overlap carried between adjacent chunks (and across block
#: boundaries on the same page/section).
CHUNK_OVERLAP_WORDS = 60

#: Chunks are committed to the database in batches of this size.
CHUNK_COMMIT_BATCH_SIZE = 100

# ---------------------------------------------------------------------------
# Semantic chunking (topic-aware splitting)
# ---------------------------------------------------------------------------
# The default chunker (split_into_sentence_chunks) cuts by WORD COUNT: it packs
# sentences until CHUNK_MAX_WORDS, then cuts — regardless of meaning. Semantic
# chunking instead embeds each sentence and cuts where adjacent sentences stop
# being similar (a topic shift), so a chunk holds one coherent idea. It costs an
# embedding pass per block at ingest time and yields variable-size chunks, so it
# is opt-in and still capped by CHUNK_MAX_WORDS as a safety net.

#: Whether chunking uses the semantic (topic-aware) splitter by default. When
#: False, the word-count splitter is used. Both honour CHUNK_MAX_WORDS; the
#: CLI (--semantic) and the Streamlit sidebar can override this per run.
SEMANTIC_CHUNKING_ENABLED = False

#: Percentile (0-100) of adjacent-sentence similarity DROPS at which to cut.
#: A document-adaptive threshold: cut where a boundary is more pronounced than
#: this share of all boundaries in the same block. Higher = fewer, larger
#: chunks; lower = more, smaller chunks. 90 cuts at the sharpest ~10% of drops.
SEMANTIC_CHUNK_PERCENTILE = 90.0

#: Absolute cosine-similarity floor used as a fallback when a block has too few
#: sentences for a meaningful percentile. Adjacent sentences below this
#: similarity are treated as a topic boundary.
SEMANTIC_CHUNK_MIN_SIMILARITY = 0.5

# ---------------------------------------------------------------------------
# Embeddings & indexing
# ---------------------------------------------------------------------------

#: sentence-transformers model used for dense embeddings (384-dim).
#: BAAI/bge-small-en-v1.5 outranks all-MiniLM-L6-v2 on retrieval (MTEB) at
#: the same dimensionality, so it is a drop-in quality upgrade. Switching
#: models requires re-embedding: `python main.py index --embed-only --reembed`.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

#: Dimensionality of the embedding model above.
EMBEDDING_DIM = 384

#: Number of chunks encoded per embedding batch.
EMBEDDING_BATCH_SIZE = 64

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

#: Default number of chunks returned by the retriever.
DEFAULT_TOP_K = 6

#: Default number of chunks retrieved for Q&A (slightly larger than
#: DEFAULT_TOP_K to give the LLM more context).
QA_TOP_K = 8

#: Default hybrid blend weight: 0 = pure BM25, 1 = pure semantic.
DEFAULT_ALPHA = 0.6

#: Reciprocal Rank Fusion constant — standard value from the original paper.
RRF_K = 60

#: Length-ratio threshold above which two retrieved chunks are treated
#: as near-duplicates (one mostly contained in the other).
DEDUP_SIMILARITY_THRESHOLD = 0.95

#: Maximum characters of retrieved context packed into an LLM prompt.
DEFAULT_CHAR_BUDGET = 12_000

# ---------------------------------------------------------------------------
# Reranking (cross-encoder)
# ---------------------------------------------------------------------------

#: Whether retrieval reranks first-stage candidates by default. The
#: retriever still accepts a per-call override.
RERANK_ENABLED = True

#: Cross-encoder model used to rerank (query, chunk) pairs. ~80 MB download;
#: scores each pair jointly rather than via independent embeddings, so it is
#: more accurate but slower than the bi-encoder used for first-stage search.
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

#: Number of first-stage candidates fetched and scored by the reranker before
#: truncating to top_k. Larger = better recall into the reranker, but slower.
RERANK_CANDIDATES = 30

#: When reranking a hybrid search, force this many top pure-semantic chunks
#: into the candidate pool even if RRF fusion would drop them (BM25 misses can
#: otherwise evict a strong semantic hit before the reranker sees it). The
#: cross-encoder then scores them fairly. Only used when rerank is on.
RERANK_GUARANTEED_SEMANTIC = 15

# ---------------------------------------------------------------------------
# Query transformation (multi-query expansion)
# ---------------------------------------------------------------------------

#: Whether retrieval expands the query into LLM-generated paraphrases by
#: default. Off by default: it adds one LLM call per query, so it is opt-in
#: (via the retriever arg or the --query-transform CLI flag) and A/B-testable.
QUERY_TRANSFORM_ENABLED = False

#: Number of additional paraphrases requested per query. The original query is
#: always retrieved alongside these, so total variants searched is N + 1.
QUERY_EXPANSION_N = 3

# ---------------------------------------------------------------------------
# Conversational memory (multi-turn chat)
# ---------------------------------------------------------------------------

#: Number of recent question/answer turns the chat loop remembers. The history
#: is injected into the generation prompt and used to rewrite follow-up
#: questions into standalone queries before retrieval.
MEMORY_MAX_TURNS = 6

#: Minimum remaining characters required to include a truncated chunk
#: at the end of a context window.
CONTEXT_MIN_TRUNCATED_CHARS = 200

# ---------------------------------------------------------------------------
# LLM providers & key rotation
# ---------------------------------------------------------------------------
# The pipeline rotates across multiple API keys / providers to survive
# free-tier daily quotas. When a key returns a quota/rate-limit error, the
# LLM layer (rag.llm) advances to the next provider in PROVIDER_CHAIN.
#
# Keys are read from environment variables. Configure as many as you have;
# missing ones are skipped. Order in PROVIDER_CHAIN is the failover order.

#: Failover order: (provider, env-var-holding-the-key, model). Groq first
#: (higher daily limits), then Gemini. Add/remove entries freely; any whose
#: env var is unset is skipped at startup.
PROVIDER_CHAIN = [
    ("groq",   "GROQ_API_KEY_1",   "llama-3.3-70b-versatile"),
    ("groq",   "GROQ_API_KEY_2",   "llama-3.3-70b-versatile"),
    ("groq",   "GROQ_API_KEY_3",   "llama-3.3-70b-versatile"),
    ("gemini", "GEMINI_API_KEY_1", "gemini-2.5-flash"),
    ("gemini", "GEMINI_API_KEY_2", "gemini-2.5-flash"),
    ("gemini", "GEMINI_API_KEY_3", "gemini-2.5-flash"),
    # Back-compat: single-key env vars used before rotation existed.
    ("groq",   "GROQ_API_KEY",     "llama-3.3-70b-versatile"),
    ("gemini", "GEMINI_API_KEY",   "gemini-2.5-flash"),
]

#: Vision failover chain: (provider, env-var, vision-capable model). Image
#: description rotates over these the same way text does. Both Groq (Llama-4
#: multimodal) and Gemini can describe images, so vision uses all 6 keys.
#: Groq first (higher daily limits). Unset keys are skipped.
VISION_PROVIDER_CHAIN = [
    ("groq",   "GROQ_API_KEY_1",   "meta-llama/llama-4-scout-17b-16e-instruct"),
    ("groq",   "GROQ_API_KEY_2",   "meta-llama/llama-4-scout-17b-16e-instruct"),
    ("groq",   "GROQ_API_KEY_3",   "meta-llama/llama-4-scout-17b-16e-instruct"),
    ("gemini", "GEMINI_API_KEY_1", "gemini-2.5-flash"),
    ("gemini", "GEMINI_API_KEY_2", "gemini-2.5-flash"),
    ("gemini", "GEMINI_API_KEY_3", "gemini-2.5-flash"),
    ("groq",   "GROQ_API_KEY",     "meta-llama/llama-4-scout-17b-16e-instruct"),
    ("gemini", "GEMINI_API_KEY",   "gemini-2.5-flash"),
]

#: Sampling temperature for Q&A / summarisation calls.
LLM_TEMPERATURE = 0.2

#: Maximum tokens for a vision description response.
VISION_MAX_TOKENS = 1024

#: Maximum characters of surrounding page text passed as context to the
#: vision model.
VISION_CONTEXT_MAX_CHARS = 1500

# ---------------------------------------------------------------------------
# Corpus-specific query routing (chat mode)
# ---------------------------------------------------------------------------
# These heuristics route chat questions to a specific document when the
# question matches a pattern. They are specific to the current corpus
# (e.g. ocr.pdf contains French user stories labelled US01..US99) —
# edit or empty this list when the corpus changes.

#: List of (regex_pattern, target_filename) pairs. If a chat query matches
#: the pattern, retrieval is restricted to the named document.
QUERY_ROUTING_RULES: list[tuple[str, str]] = [
    (r"\bUS\d{2}\b", "ocr.pdf"),
    (r"\b(en tant que|livreur|client|commande|livraison|priorit[eé]|parcourir)\b", "ocr.pdf"),
]
