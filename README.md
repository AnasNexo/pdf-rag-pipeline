# PDF RAG Pipeline

## Overview

A local Retrieval-Augmented Generation (RAG) system over PDF documents. It parses PDFs (including scanned ones), stores everything in a single SQLite file, builds hybrid search indexes, and answers questions or produces summaries with an LLM — with inline source citations.

High-level architecture:

```
        PDF files
            │
  ┌─────────▼──────────┐   PyMuPDF text layer, RapidOCR fallback,
  │   1. INGEST        │   pdfplumber/Camelot tables,
  │  (rag/pipeline.py) │   LLM vision image descriptions
  └─────────┬──────────┘
            │  documents / pages / images / tables / semantic_blocks
  ┌─────────▼──────────┐
  │   2. CHUNK         │   word-count OR semantic (topic-aware) splitting,
  │  (rag/chunker.py)  │   tables & image descriptions kept atomic
  └─────────┬──────────┘
            │  chunks
  ┌─────────▼──────────┐
  │   3. INDEX         │   BM25 via SQLite FTS5 (Porter stemming)
  │  (rag/indexer.py)  │   dense vectors via sentence-transformers
  └─────────┬──────────┘
            │  chunks_fts / chunk_embeddings
  ┌─────────▼──────────┐
  │   4. RETRIEVE      │   bm25 / semantic / hybrid (RRF fusion),
  │ (rag/retriever.py) │   optional query expansion + cross-encoder rerank,
  └─────────┬──────────┘   dedup + citation-aware context packing
            │  prompt context
  ┌─────────▼──────────┐
  │   5. GENERATE      │   Groq/Gemini (rotated on quota), strictly grounded
  │ (rag/generator.py) │   Q&A and summarisation with citations
  └────────────────────┘
```

Everything lives in one SQLite database (`pdf_data.db` by default) — no external vector store or services required. The only network dependency is the LLM API (image descriptions during ingestion, chat completions during Q&A). The LLM layer **rotates across multiple Groq and Gemini API keys**, advancing to the next when one hits its free-tier quota.

Two front-ends share the same pipeline:

- **CLI** (`main.py`) — `ingest / chunk / index / search / ask / chat / summarise / ...`
- **Web app** (`app.py`, Streamlit) — a chat UI with conversation history, PDF upload, and live-streamed sourced answers.

## Project Structure

```
.
├── main.py                # Unified CLI: ingest / chunk / index / search / ask / chat / ...
├── app.py                 # Streamlit chat interface (upload, history, streaming answers)
├── config.py              # ALL tunable constants (paths, thresholds, models, provider chains)
├── requirements.txt
├── .env.example           # Names of the API-key environment variables (copy to .env)
├── README.md
├── tests_smoke.py         # Dependency-free sanity checks (no keys needed)
├── LICENSE                # MIT
├── Dockerfile             # Container image (bundles Ghostscript + libGL)
├── docker-compose.yml     # One-command run; named volumes for DB + model cache
├── .dockerignore
├── .streamlit/
│   └── config.toml        # Streamlit theme / server settings
├── data/                  # Sample PDF documents
├── pdf_data.db            # SQLite database (created by ingest; not committed)
├── eval/                  # Evaluation harness + saved experiment reports
│   ├── README.md          # How to run the harness and read the reports
│   ├── evaluate.py        # hit@k / recall@k / MRR + LLM-as-judge answer grading
│   ├── eval_set.jsonl     # Hand-written questions with ground-truth signals
│   ├── eval_set_vague.jsonl  # Under-specified questions (query-expansion probe)
│   └── report_*.json      # Saved results, one per experiment configuration
└── rag/
    ├── __init__.py        # Package overview
    ├── db.py              # SQLite schema, migrations, connections, deletion, diagnostics
    ├── utils.py           # Shared helpers (file hashing, text stats, bbox formatting)
    ├── loader.py          # Page analysis: text-layer vs OCR decision, semantic blocks,
    │                      #   heading detection, page rendering, visual-content detection
    ├── ocr.py             # RapidOCR wrapper + text-dominance heuristics
    ├── tables.py          # Table extraction (pdfplumber grid/rects + Camelot lattice/stream)
    ├── vision.py          # Image/chart/diagram descriptions (delegates to rag.llm)
    ├── llm.py             # Multi-provider LLM layer: Groq + Gemini with key rotation
    ├── pipeline.py        # Ingestion orchestration (process_pdf / process_page)
    ├── chunker.py         # Word-count and semantic (topic-aware) chunking
    ├── embedder.py        # sentence-transformers cache + vector (de)serialisation
    ├── indexer.py         # FTS5 + embedding index builders (incremental, idempotent)
    ├── search.py          # bm25_search / semantic_search / hybrid_search (RRF)
    ├── reranker.py        # Cross-encoder reranking of first-stage candidates
    ├── query_transform.py # Multi-query expansion + follow-up rewriting (RRF fusion)
    ├── retriever.py       # Retriever class: ranked retrieval, dedup, context packing
    ├── generator.py       # LLM Q&A + summarisation (prompts, ask(), summarise())
    ├── memory.py          # Bounded conversation memory for multi-turn chat
    ├── history.py         # Persistent saved conversations (Streamlit sidebar)
    ├── manage.py          # UI helpers: add PDFs, re-chunk in place, reset corpus
    └── chat.py            # Interactive CLI chat loop + meta-query routing
```

Why this split:

- **`config.py` at the root** — every threshold, model name, and default in one importable place; nothing is hardcoded inside the stages.
- **`db.py` owns the full schema** — the whole data model is readable in one file, and deletion/migration logic lives next to it.
- **One module per pipeline stage** — each stage can be run, tested, and reasoned about independently; modules only import "downward" (e.g. `generator` → `retriever` → `search` → `embedder`).
- **Heavy imports stay lazy** — RapidOCR, sentence-transformers, and the LLM SDKs are loaded on first use, so e.g. `main.py diagnose` starts instantly.

## Setup & Installation

**Requires Python 3.10+** (tested on 3.10 and, via Docker, 3.11).

1. **Clone the project**, then create and activate a virtual environment:

   ```powershell
   cd path\to\project
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   # source .venv/bin/activate     # macOS / Linux
   ```

2. **Install dependencies:**

   ```powershell
   pip install -r requirements.txt
   ```

   Note: `camelot-py[cv]` also requires **Ghostscript** on some systems, and `opencv-python-headless` requires system libraries (`libGL`) on slim Linux. `sentence-transformers` downloads the embedding + reranker models (~100 MB total) on first use.

3. **Set your API keys.** The pipeline rotates across every key that is set (Groq first, then Gemini); the minimum is one key.

   The simplest way is a `.env` file in the project root — it is loaded
   automatically on startup, and `.gitignore` already excludes it:

   ```powershell
   cp .env.example .env     # then open it and fill in at least one key
   ```

   Real environment variables also work and take precedence over `.env`:

   ```powershell
   $env:GROQ_API_KEY_1 = "your-key-here"     # PowerShell
   # export GROQ_API_KEY_1="your-key-here"   # bash
   ```

   See `.env.example` for the full list of variable names.

4. **Put your PDFs in `data/`** (or anywhere — paths are passed explicitly).

## Configuration

All knobs live in `config.py`. Key parameters:

| Key | Default | Meaning |
|---|---|---|
| `DEFAULT_DB_PATH` | `pdf_data.db` | SQLite file used by every stage. |
| `TEXT_LAYER_WORD_THRESHOLD` / `TEXT_LAYER_CHAR_THRESHOLD` | 30 / 180 | Minimum words/chars for a PDF text layer to be trusted; below this the page is OCR'd. |
| `PAGE_RENDER_DPI` | 200 | Rasterisation resolution for OCR/vision. |
| `CHUNK_MAX_WORDS` / `CHUNK_OVERLAP_WORDS` | 300 / 60 | Retrieval chunk size and overlap. |
| `SEMANTIC_CHUNKING_ENABLED` | `False` | Use topic-aware chunking (embed sentences, cut on similarity drops) instead of word-count. `max_words` still caps chunk size. |
| `SEMANTIC_CHUNK_PERCENTILE` | 90 | Document-adaptive cut threshold for semantic chunking. |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | `BAAI/bge-small-en-v1.5` / 384 | Dense embedding model. Changing it requires re-embedding (`index --reembed`). |
| `DEFAULT_TOP_K` / `QA_TOP_K` | 6 / 8 | Chunks returned by retrieval / sent to the LLM. |
| `DEFAULT_ALPHA` | 0.6 | Hybrid blend: 0 = pure BM25, 1 = pure semantic. |
| `RRF_K` | 60 | Reciprocal Rank Fusion constant (standard value). |
| `RERANK_ENABLED` / `RERANK_MODEL` | `True` / `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranking of first-stage candidates. |
| `RERANK_CANDIDATES` | 30 | First-stage candidates fetched and scored by the reranker. |
| `QUERY_TRANSFORM_ENABLED` / `QUERY_EXPANSION_N` | `False` / 3 | Multi-query expansion (paraphrase the query, fuse results). Off by default (one extra LLM call). |
| `MEMORY_MAX_TURNS` | 6 | Conversation turns remembered in multi-turn chat. |
| `DEFAULT_CHAR_BUDGET` | 12000 | Max characters of context packed into an LLM prompt. |
| `PROVIDER_CHAIN` / `VISION_PROVIDER_CHAIN` | see file | Failover order of `(provider, env-var, model)` for text and vision. Unset keys are skipped. |
| `LLM_TEMPERATURE` | 0.2 | Generation temperature for Q&A / summaries. |
| `QUERY_ROUTING_RULES` | see file | **Corpus-specific** regex → filename rules that route chat questions to a document. Edit or empty when your corpus changes. |

Most CLI subcommands accept flags (`--top-k`, `--alpha`, `--semantic`, `--no-rerank`, `--query-transform`, ...) that override these defaults per run.

## Usage

### Web app (recommended)

```powershell
streamlit run app.py
```

Upload PDFs from the sidebar (choose **word-count** or **semantic** chunking), then chat. You can re-chunk the existing corpus with a different method in place (no re-upload), browse saved conversations, and adjust retrieval knobs (mode, query expansion, top_k).

### Docker

The image bundles the system dependencies that are otherwise easy to miss
(Ghostscript for Camelot, `libGL` for OpenCV), so it "just runs":

```bash
cp .env.example .env          # then fill in at least one API key
docker compose up --build     # open http://localhost:8501
```

The ingested corpus DB and the downloaded models are kept in named volumes, so
they survive restarts. Without compose:

```bash
docker build -t pdf-rag .
docker run -p 8501:8501 --env-file .env -v rag_db:/app/db \
  -e RAG_DB_PATH=/app/db/pdf_data.db pdf-rag
```

### CLI — end-to-end pipeline

```powershell
# 1. Ingest PDFs (re-runs skip unchanged files by content hash; --force reprocesses)
python main.py ingest data\test.pdf data\rep.pdf data\ocr.pdf data\im.pdf

# 2. Build chunks (idempotent; --reset rebuilds; --semantic for topic-aware)
python main.py chunk
python main.py chunk --reset --semantic          # switch to semantic chunking

# 3. Build search indexes (incremental: only new chunks are indexed/embedded)
python main.py index

# 4a. Interactive chat
python main.py chat

# 4b. One-shot question (--query-transform / --no-rerank to toggle stages)
python main.py ask "What are the side effects?"

# 4c. Summarise a document or a section
python main.py summarise --document-id 1
python main.py summarise --document-id 1 --section "Introduction"
```

### Inspection and maintenance

```powershell
python main.py search "frequency response" --mode bm25 --top-k 8
python main.py list                       # all documents
python main.py list --document-id 1       # sections of document 1
python main.py diagnose                   # row counts + orphan check
python main.py delete rep.pdf             # remove a document and all derived data
```

In the chat loop you can also ask meta-questions answered straight from the database — "how many documents do you have?", "which documents are in French?", "summarise doc 2" — without an LLM round-trip (summaries do call the LLM).

## Pipeline Walkthrough

1. **Load (`rag/pipeline.py` + `rag/loader.py`)** — each PDF is hashed (SHA-256) for dedup, then processed page by page inside SQLite savepoints (one bad page can't lose the document). Pages with a substantial text layer keep their extracted text and are split into typed *semantic blocks* (headings detected by relative font size; the current heading is attached to following blocks as `section_title`). Sparse/scanned pages are rendered at 200 DPI and OCR'd with RapidOCR.

2. **Tables (`rag/tables.py`)** — four detection strategies run in confidence order (pdfplumber line-grid, pdfplumber rects, Camelot lattice, Camelot stream gated by its accuracy score), validated against paragraph-text false positives, deduplicated by content hash, and stored as both markdown and raw cell JSON.

3. **Images (`rag/ocr.py` + `rag/vision.py`)** — each embedded image (and whole pages that are scanned or visually rich) is OCR'd first; if text-dominant, the OCR text becomes the description for free. Otherwise the image goes to the LLM vision model (Groq Llama-4, then Gemini) with surrounding page text as context.

4. **Chunk (`rag/chunker.py`)** — semantic blocks become retrieval chunks. The default splitter packs sentences up to 300 words with 60 words of overlap (overlap resets at page/section boundaries). The **semantic** splitter instead embeds each sentence and cuts where adjacent sentences stop being similar (a topic shift), still capped by `max_words`. Tables and image descriptions stay atomic.

5. **Embed & store (`rag/embedder.py` + `rag/indexer.py`)** — chunks are indexed twice: into a SQLite FTS5 table with Porter stemming (BM25), and as L2-normalised 384-dim `bge-small-en-v1.5` vectors stored as float32 blobs. Both builders are incremental.

6. **Retrieve (`rag/search.py`, `rag/retriever.py`, `rag/reranker.py`, `rag/query_transform.py`)** — hybrid search runs BM25 and cosine-similarity search, then fuses rankings with Reciprocal Rank Fusion. Optionally the query is first expanded into paraphrases (multi-query). Candidates are deduplicated, then rescored by a cross-encoder reranker, and the survivors are packed into a citation-annotated context string within a character budget.

7. **Generate (`rag/generator.py` + `rag/llm.py`)** — the context plus the question go to the active LLM (Groq Llama 3.3 70B, rotating to Gemini on quota) with a strict grounding prompt: answer only from the excerpts, cite `(filename, p.N)` inline, and explicitly say when the documents don't contain the answer (including the case where only an image description exists).

## Smoke tests

A dependency-free check that a clone is wired up correctly — every module
imports, the config and provider chains are consistent, no key material is
committed, and an ingest-free chunking round-trip plus the FTS5 index work.
No API keys and no network required:

```powershell
python tests_smoke.py
```

## Evaluation

`eval/evaluate.py` measures the pipeline against a hand-written eval set (`eval/eval_set.jsonl`):

- **Retrieval quality** (deterministic, no LLM cost): `hit@k`, `recall@k`, `MRR`.
- **Answer quality** (LLM-as-judge): faithfulness, relevance, correctness (1–5).

```powershell
python eval/evaluate.py --report eval/report.json
python eval/evaluate.py --no-judge                 # retrieval metrics only
python eval/evaluate.py --semantic --report eval/report_semantic.json   # A/B a variant
```

Saved `eval/report_*.json` files record past experiments (embedding model, reranking, fusion fix, query transform, semantic vs word-count chunking).

## Dependencies

| Library | Role |
|---|---|
| **pymupdf** (`fitz`) | Fast PDF parsing: text layer, font/span info for heading detection, embedded image extraction, page rendering. |
| **pdfplumber** | Line/rect geometry used for table detection strategies 1–2. |
| **camelot-py[cv]** + **opencv-python-headless** | Lattice and stream table extraction (strategies 3–4) with built-in accuracy scoring. |
| **rapidocr_onnxruntime** | Local, dependency-light OCR for scanned pages and images (no cloud calls). |
| **pillow** / **numpy** | Image decoding for OCR; vector math for cosine search. |
| **sentence-transformers** | Dense embeddings (`bge-small-en-v1.5`) and the cross-encoder reranker. |
| **groq** / **google-genai** | API clients for the vision model (image descriptions) and the chat model (Q&A/summaries), used with key rotation. |
| **python-dotenv** | Loads API keys from a local `.env` file at startup (optional; real env vars take precedence). |
| **streamlit** | Web chat interface. |
| **sqlite3** (stdlib) | Single-file storage for documents, blocks, chunks, FTS5 BM25 index, and embedding blobs — no external vector DB needed. |

## License

MIT — see [LICENSE](LICENSE).
