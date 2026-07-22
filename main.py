"""Unified CLI entry point for the RAG pipeline.

Subcommands map one-to-one onto pipeline stages:

    ingest    — parse PDFs into SQLite (text, OCR, tables, image descriptions)
    chunk     — build retrieval chunks from semantic blocks
    index     — build BM25 (FTS5) and dense-embedding indexes
    search    — run a test search against the indexes
    ask       — answer a single question with the LLM
    chat      — interactive Q&A loop
    summarise — summarise a document or section with the LLM
    list      — list documents (or sections of one document)
    diagnose  — print DB contents and check for orphaned rows
    delete    — remove a document and all derived data

Typical end-to-end run:

    set GROQ_API_KEY_1=your-key-here   (see .env.example for all key names)
    python main.py ingest data\\test.pdf data\\rep.pdf
    python main.py chunk
    python main.py index
    python main.py chat
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import config
from rag import db
from rag.chunker import ChunkConfig, chunk_database


def _add_db_arg(parser: argparse.ArgumentParser) -> None:
    """Attach the shared --db option to a subparser.

    Args:
        parser (argparse.ArgumentParser): Subparser to extend.

    Returns:
        None
    """
    parser.add_argument("--db", default=config.DEFAULT_DB_PATH,
                        help="SQLite database path")


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser with all subcommands.

    Returns:
        argparse.ArgumentParser: Fully configured parser.
    """
    parser = argparse.ArgumentParser(
        description="PDF RAG pipeline: ingest → chunk → index → retrieve → generate"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="Parse PDFs into the database")
    _add_db_arg(p)
    p.add_argument("pdfs", nargs="+", help="Path(s) to PDF file(s)")
    p.add_argument("--force", action="store_true",
                   help="Reprocess even if the content hash already exists")

    p = sub.add_parser("chunk", help="Build retrieval chunks from semantic blocks")
    _add_db_arg(p)
    p.add_argument("--max-words", type=int, default=config.CHUNK_MAX_WORDS,
                   help="Maximum words per chunk")
    p.add_argument("--overlap-words", type=int, default=config.CHUNK_OVERLAP_WORDS,
                   help="Overlapping words between adjacent chunks")
    p.add_argument("--reset", action="store_true",
                   help="Delete existing chunks before rebuilding")
    p.add_argument("--document-id", type=int, default=None,
                   help="Scope reset and chunking to a single document ID")
    p.add_argument("--semantic", dest="semantic", action="store_true",
                   default=config.SEMANTIC_CHUNKING_ENABLED,
                   help="Cut chunks at topic shifts (embed sentences, split on "
                        "similarity drops) instead of purely by word count")
    p.add_argument("--word-count", dest="semantic", action="store_false",
                   help="Force the word-count splitter (overrides config)")
    p.add_argument("--semantic-percentile", type=float,
                   default=config.SEMANTIC_CHUNK_PERCENTILE,
                   help="Percentile of sentence-similarity drops to cut at "
                        "(higher = fewer, larger semantic chunks)")

    p = sub.add_parser("index", help="Build BM25 and embedding indexes")
    _add_db_arg(p)
    p.add_argument("--fts-only", action="store_true", help="Build BM25 FTS index only")
    p.add_argument("--embed-only", action="store_true",
                   help="Build dense embeddings only")
    p.add_argument("--reembed", action="store_true",
                   help="Drop embeddings from other models and rebuild "
                        "(use after changing --model / EMBEDDING_MODEL)")
    p.add_argument("--model", default=config.EMBEDDING_MODEL,
                   help="Sentence-transformer model name")

    p = sub.add_parser("search", help="Run a test search against the indexes")
    _add_db_arg(p)
    p.add_argument("query", help="Search query")
    p.add_argument("--mode", choices=["bm25", "semantic", "hybrid"],
                   default="hybrid", help="Search mode")
    p.add_argument("--top-k", type=int, default=5, help="Number of results")
    p.add_argument("--alpha", type=float, default=config.DEFAULT_ALPHA,
                   help="Hybrid blend: 0=BM25 only, 1=semantic only")
    p.add_argument("--model", default=config.EMBEDDING_MODEL,
                   help="Sentence-transformer model name")

    p = sub.add_parser("ask", help="Answer a single question with the LLM")
    _add_db_arg(p)
    p.add_argument("query", help="The question to answer")
    p.add_argument("--mode", choices=["hybrid", "bm25", "semantic"], default="hybrid")
    p.add_argument("--alpha", type=float, default=config.DEFAULT_ALPHA)
    p.add_argument("--top-k", type=int, default=config.QA_TOP_K)
    p.add_argument("--char-budget", type=int, default=config.DEFAULT_CHAR_BUDGET,
                   help="Max characters of context sent to the LLM")
    p.add_argument("--model", default=config.EMBEDDING_MODEL,
                   help="Sentence-transformer model for retrieval")
    p.add_argument("--no-rerank", dest="rerank", action="store_false",
                   default=config.RERANK_ENABLED,
                   help="Disable cross-encoder reranking of retrieved chunks")
    p.add_argument("--query-transform", dest="query_transform",
                   action="store_true", default=config.QUERY_TRANSFORM_ENABLED,
                   help="Expand the query into LLM-generated paraphrases "
                        "(multi-query retrieval)")

    p = sub.add_parser("chat", help="Interactive Q&A loop")
    _add_db_arg(p)
    p.add_argument("--mode", choices=["hybrid", "bm25", "semantic"], default="hybrid")
    p.add_argument("--alpha", type=float, default=config.DEFAULT_ALPHA)
    p.add_argument("--top-k", type=int, default=config.QA_TOP_K)
    p.add_argument("--char-budget", type=int, default=config.DEFAULT_CHAR_BUDGET)
    p.add_argument("--model", default=config.EMBEDDING_MODEL,
                   help="Sentence-transformer model for retrieval")
    p.add_argument("--no-rerank", dest="rerank", action="store_false",
                   default=config.RERANK_ENABLED,
                   help="Disable cross-encoder reranking of retrieved chunks")
    p.add_argument("--query-transform", dest="query_transform",
                   action="store_true", default=config.QUERY_TRANSFORM_ENABLED,
                   help="Expand the query into LLM-generated paraphrases "
                        "(multi-query retrieval)")

    p = sub.add_parser("summarise", help="Summarise a document or section",
                       aliases=["summarize"])
    _add_db_arg(p)
    p.add_argument("--document-id", type=int, required=True)
    p.add_argument("--section", help="Section title to summarise")
    p.add_argument("--char-budget", type=int, default=config.DEFAULT_CHAR_BUDGET)
    p.add_argument("--model", default=config.EMBEDDING_MODEL,
                   help="Sentence-transformer model for retrieval")

    p = sub.add_parser("list", help="List documents, or sections of one document")
    _add_db_arg(p)
    p.add_argument("--document-id", type=int,
                   help="List sections of this document instead of documents")

    p = sub.add_parser("diagnose", help="Print DB contents and check for orphans")
    _add_db_arg(p)

    p = sub.add_parser("delete", help="Remove a document and all its data")
    _add_db_arg(p)
    p.add_argument("ref", metavar="FILENAME_OR_ID",
                   help="Document filename or numeric ID")

    return parser


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def cmd_ingest(args: argparse.Namespace) -> int:
    """Run the ingestion pipeline over the given PDFs.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        int: Process exit code (0 on success).
    """
    from rag.pipeline import process_pdf

    conn = db.connect(args.db)
    db.init_schema(conn)

    for pdf_path in args.pdfs:
        if not pdf_path.lower().endswith(".pdf"):
            print(f"Skipping '{pdf_path}': does not look like a PDF.", file=sys.stderr)
            continue
        if not os.path.isfile(pdf_path):
            print(f"Skipping '{pdf_path}': file not found", file=sys.stderr)
            continue
        try:
            process_pdf(pdf_path, conn, force=args.force)
        except Exception as e:
            print(f"Error processing '{pdf_path}': {e}", file=sys.stderr)
            traceback.print_exc()

    conn.close()
    print(f"All done. Data stored in '{args.db}'.")
    return 0


def cmd_chunk(args: argparse.Namespace) -> int:
    """Build retrieval chunks from semantic blocks.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        int: Process exit code (0 on success).
    """
    chunk_config = ChunkConfig(
        max_words=args.max_words,
        overlap_words=args.overlap_words,
        reset=args.reset,
        document_id=args.document_id,
        semantic=args.semantic,
        semantic_percentile=args.semantic_percentile,
    )
    conn = db.connect(args.db, must_exist=True)
    try:
        inserted = chunk_database(conn, chunk_config)
        print(f"Built {inserted} chunk(s) for '{args.db}' "
              "(already-existing chunks are ignored).")
    finally:
        conn.close()
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    """Build the BM25 and/or dense-embedding indexes.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        int: Process exit code (0 on success).
    """
    from rag.indexer import build_embeddings, build_fts_index

    conn = db.connect(args.db, must_exist=True)
    try:
        if not args.embed_only:
            print("Building FTS5 BM25 index...")
            n = build_fts_index(conn)
            print(f"  FTS index: {n} rows total.")
        if not args.fts_only:
            print("Building dense embeddings...")
            n = build_embeddings(conn, model_name=args.model,
                                 reembed=args.reembed)
            print(f"  Newly embedded: {n} chunks.")
        print("Done.")
    finally:
        conn.close()
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Run a test search and print the ranked results.

    Builds any missing index on the fly before searching.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        int: Process exit code (0 on success).
    """
    import json
    import sqlite3

    from rag.indexer import build_embeddings, build_fts_index
    from rag.search import bm25_search, hybrid_search, semantic_search

    conn = db.connect(args.db, must_exist=True)
    try:
        if args.mode in ("bm25", "hybrid"):
            try:
                n_fts = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
            except sqlite3.OperationalError:
                n_fts = 0
            if not n_fts:
                print("FTS index empty or not found — building now...")
                build_fts_index(conn)

        if args.mode in ("semantic", "hybrid"):
            try:
                n_emb = conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0]
            except sqlite3.OperationalError:
                n_emb = 0
            if not n_emb:
                print("Embedding table empty or not found — building now...")
                build_embeddings(conn, model_name=args.model)

        print(f"\nSearching [{args.mode}]: '{args.query}'\n")
        if args.mode == "bm25":
            results = bm25_search(conn, args.query, top_k=args.top_k)
        elif args.mode == "semantic":
            results = semantic_search(conn, args.query, top_k=args.top_k,
                                      model_name=args.model)
        else:
            results = hybrid_search(conn, args.query, top_k=args.top_k,
                                    model_name=args.model, alpha=args.alpha)

        if not results:
            print("No results found.")
            return 0
        # Plain ASCII separator: box-drawing chars crash on cp1252 consoles.
        for i, r in enumerate(results, 1):
            meta = json.loads(r.get("metadata_json") or "{}")
            section = meta.get("section_title") or ""
            section_str = f"  section='{section}'" if section else ""
            print(f"\n{'-' * 64}")
            print(f"#{i}  score={r['score']:.4f}  "
                  f"page={r.get('page_number')}  "
                  f"file={r.get('filename')}  "
                  f"type={r.get('chunk_type')}"
                  f"{section_str}")
            print(r["text_content"][:500])
        print(f"\n{'-' * 64}")
    finally:
        conn.close()
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    """Answer a single question and print the sources used.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        int: Process exit code (0 on success).
    """
    from rag.chat import print_sources
    from rag.generator import ask
    from rag.retriever import Retriever

    with Retriever(args.db, model_name=args.model, default_mode=args.mode,
                   default_alpha=args.alpha, rerank=args.rerank,
                   query_transform=args.query_transform) as retriever:
        _answer, chunks = ask(
            retriever, args.query,
            top_k=args.top_k, char_budget=args.char_budget,
            mode=args.mode, alpha=args.alpha, stream=True,
        )
        print_sources(chunks)
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    """Start the interactive Q&A loop.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        int: Process exit code (0 on success).
    """
    from rag.chat import chat_loop
    from rag.retriever import Retriever

    with Retriever(args.db, model_name=args.model, default_mode=args.mode,
                   default_alpha=args.alpha, rerank=args.rerank,
                   query_transform=args.query_transform) as retriever:
        chat_loop(retriever, top_k=args.top_k, char_budget=args.char_budget,
                  mode=args.mode, alpha=args.alpha)
    return 0


def cmd_summarise(args: argparse.Namespace) -> int:
    """Summarise a document or one of its sections.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        int: Process exit code (0 on success).
    """
    from rag.generator import summarise
    from rag.retriever import Retriever

    with Retriever(args.db, model_name=args.model) as retriever:
        summarise(retriever, args.document_id, section=args.section,
                  char_budget=args.char_budget, stream=True)
        print()
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List documents, or the sections of one document.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        int: Process exit code (0 on success).
    """
    conn = db.connect(args.db, must_exist=True)
    try:
        if args.document_id:
            cur = conn.execute(
                """
                SELECT DISTINCT json_extract(c.metadata_json, '$.section_title') AS sec
                FROM chunks c
                WHERE c.document_id = ? AND sec IS NOT NULL
                ORDER BY c.id
                """,
                (args.document_id,),
            )
            sections = [r["sec"] for r in cur]
            if not sections:
                print("No sections found.")
            for s in sections:
                print(f"  {s}")
        else:
            docs = db.list_documents(conn)
            if not docs:
                print("No documents found.")
            for d in docs:
                print(f"  id={d['id']}  {d['filename']}")
    finally:
        conn.close()
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    """Print database contents and check for orphaned rows.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        int: Process exit code (0 on success).
    """
    db.diagnose(args.db)
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    """Delete a document and all its derived data.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        int: Process exit code (0 on success, 1 if not found).
    """
    if not os.path.isfile(args.db):
        print(f"Database not found: {args.db}", file=sys.stderr)
        return 1
    conn = db.connect(args.db)
    try:
        doc = db.find_document(conn, args.ref)
        if not doc:
            known = [d["filename"] for d in db.list_documents(conn)]
            print(f"Document '{args.ref}' not found. Known: {', '.join(known)}",
                  file=sys.stderr)
            return 1
        db.delete_document(conn, doc["id"])
        print(f"Deleted document '{doc['filename']}' (id={doc['id']}) "
              "and all its derived data.")
    finally:
        conn.close()
    return 0


_COMMANDS = {
    "ingest": cmd_ingest,
    "chunk": cmd_chunk,
    "index": cmd_index,
    "search": cmd_search,
    "ask": cmd_ask,
    "chat": cmd_chat,
    "summarise": cmd_summarise,
    "summarize": cmd_summarise,
    "list": cmd_list,
    "diagnose": cmd_diagnose,
    "delete": cmd_delete,
}


def main() -> int:
    """Parse arguments and dispatch to the selected subcommand.

    Returns:
        int: Process exit code.
    """
    args = build_parser().parse_args()
    return _COMMANDS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
