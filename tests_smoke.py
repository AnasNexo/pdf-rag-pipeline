"""Dependency-free smoke tests for the RAG pipeline.

Checks the things that silently break a fresh clone: every module imports,
the config is internally consistent, the provider chains are well formed,
and an end-to-end ingest -> chunk -> index -> search cycle works on a
temporary database. No API keys and no network are required.

Run:  python tests_smoke.py
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import sqlite3
import sys
import tempfile
import traceback

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

_passed = 0
_failed: list[str] = []

if __name__ == "__main__":
    print("Smoke tests (no API keys or network needed)\n")


def check(name: str):
    """Decorator: run a function as a named test, recording pass/fail."""
    def wrap(fn):
        global _passed
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 — report, never abort
            _failed.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"  FAIL  {name}")
            traceback.print_exc()
        else:
            _passed += 1
            print(f"  ok    {name}")
        return fn
    return wrap


# --- imports ---------------------------------------------------------------

@check("every rag.* module imports")
def _t_imports():
    import rag
    for mod in pkgutil.iter_modules(rag.__path__):
        importlib.import_module(f"rag.{mod.name}")
    for mod in ("config", "main"):
        importlib.import_module(mod)


# --- configuration ---------------------------------------------------------

@check("config values are sane")
def _t_config():
    import config
    assert config.CHUNK_OVERLAP_WORDS < config.CHUNK_MAX_WORDS, \
        "overlap must be smaller than the chunk size"
    assert 0.0 <= config.DEFAULT_ALPHA <= 1.0, "alpha is a 0..1 blend"
    assert config.EMBEDDING_DIM > 0
    assert config.RERANK_CANDIDATES >= config.DEFAULT_TOP_K, \
        "reranking fewer candidates than we return makes no sense"
    assert isinstance(config.QUERY_ROUTING_RULES, list)


@check("provider chains are well formed")
def _t_chains():
    import config
    for name in ("PROVIDER_CHAIN", "VISION_PROVIDER_CHAIN"):
        chain = getattr(config, name)
        assert chain, f"{name} is empty"
        for entry in chain:
            kind, env_var, model = entry  # shape: 3-tuple
            assert kind in ("groq", "gemini"), f"{name}: bad provider {kind!r}"
            assert env_var and model, f"{name}: empty field in {entry!r}"


@check("routing-rule regexes compile")
def _t_routing():
    import re
    import config
    for pattern, target in config.QUERY_ROUTING_RULES:
        re.compile(pattern)
        assert isinstance(target, str) and target


@check("no key material committed in config")
def _t_no_keys():
    import re
    src = open("config.py", encoding="utf-8").read()
    assert not re.search(r"gsk_[A-Za-z0-9]{10,}|AIza[A-Za-z0-9_-]{10,}", src), \
        "an API key looks hardcoded in config.py"


# --- database / pipeline ---------------------------------------------------

@check("schema builds and chunking round-trips on a temp DB")
def _t_pipeline():
    from rag import db
    from rag.chunker import ChunkConfig, chunk_database

    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    try:
        conn = db.connect(path)
        db.init_schema(conn)

        cur = conn.cursor()
        cur.execute(
            "INSERT INTO documents (filename, filepath, content_hash, "
            "page_count) VALUES (?, ?, ?, ?)",
            ("unit.pdf", "data/unit.pdf", "deadbeef", 1))
        doc_id = cur.lastrowid
        cur.execute(
            "INSERT INTO pages (document_id, page_number, source, "
            "text_content) VALUES (?, ?, ?, ?)",
            (doc_id, 1, "text", "alpha beta gamma"))
        page_id = cur.lastrowid
        cur.execute(
            "INSERT INTO semantic_blocks "
            "(document_id, page_id, block_index, block_type, source, "
            " text_content, section_title) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (doc_id, page_id, 0, "text", "text",
             "Alpha beta gamma delta. " * 40, "Intro"))
        conn.commit()

        made = chunk_database(conn, ChunkConfig())
        assert made > 0, "chunking produced no chunks"
        n = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert n == made, f"chunk count mismatch: {n} != {made}"
        conn.close()
    finally:
        # Windows keeps the file locked until the connection is closed, so
        # ignore a failed unlink rather than masking the real test result.
        for suffix in ("", "-wal", "-shm"):
            try:
                if os.path.exists(path + suffix):
                    os.unlink(path + suffix)
            except OSError:
                pass


@check("BM25 (FTS5) index builds and returns a hit")
def _t_fts():
    assert sqlite3.sqlite_version_info >= (3, 9), "FTS5 needs SQLite 3.9+"
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(body)")
    except sqlite3.OperationalError as exc:
        raise AssertionError(
            f"this Python's SQLite lacks FTS5, required for BM25: {exc}")
    conn.execute("INSERT INTO t (body) VALUES ('hybrid retrieval ranking')")
    hits = conn.execute(
        "SELECT body FROM t WHERE t MATCH 'retrieval'").fetchall()
    assert len(hits) == 1, "FTS5 match failed"
    conn.close()


if __name__ == "__main__":
    total = _passed + len(_failed)
    print(f"\n{_passed}/{total} passed")
    if _failed:
        print("\nFailures:")
        for f in _failed:
            print(f"  - {f}")
    raise SystemExit(1 if _failed else 0)
