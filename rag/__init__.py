"""RAG pipeline package.

Stages (each in its own module):

    pipeline   — PDF ingestion: text extraction, OCR, tables, image descriptions
    chunker    — word-count and semantic (topic-aware) chunking of blocks
    indexer    — BM25 (SQLite FTS5) and dense-embedding index construction
    search          — bm25 / semantic / hybrid search primitives
    query_transform — multi-query expansion (LLM paraphrase + RRF fusion)
    reranker        — cross-encoder reranking of first-stage candidates
    retriever       — high-level retrieval API (expand, dedup, rerank, pack)
    generator  — LLM Q&A and summarisation (via rag.llm)
    memory     — bounded multi-turn conversation memory for chat
    chat       — interactive chat loop with meta-query routing + memory

Supporting modules:

    db         — SQLite schema, migrations, connections, deletion, diagnostics
    loader     — page-level text/visual analysis and semantic-block extraction
    ocr        — RapidOCR wrapper
    tables     — table extraction (pdfplumber + Camelot)
    vision     — multimodal image description (delegates to rag.llm)
    llm        — multi-provider LLM layer (Groq + Gemini) with key rotation
    embedder   — sentence-transformers model cache and vector (de)serialisation
    history    — persistent saved conversations (Streamlit UI)
    manage     — corpus management helpers for the web UI
    utils      — small shared helpers
"""
