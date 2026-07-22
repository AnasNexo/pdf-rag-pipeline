"""Persistent chat history: multiple saved conversations in SQLite.

Stores conversations and their messages in the same database file as the
document corpus (config.DEFAULT_DB_PATH), in two dedicated tables that are
independent of the ingestion schema:

    conversations(id, title, created_at, updated_at)
    chat_messages(id, conversation_id, role, content, sources, created_at)

The UI uses this to list past conversations, load one back into the chat,
start a new one, and delete one — a ChatGPT-style history sidebar. Each
message's source citations are stored as a JSON list so they re-render on
reload.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from rag import db


def _now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: str) -> sqlite3.Connection:
    """Open the database (cross-thread) and ensure the history schema exists.

    Args:
        db_path (str): SQLite database path.

    Returns:
        sqlite3.Connection: Open connection with history tables present.
    """
    conn = db.connect(db_path, same_thread=False)
    init_history_schema(conn)
    return conn


def init_history_schema(conn: sqlite3.Connection) -> None:
    """Create the chat-history tables if they do not exist.

    Safe to call repeatedly.

    Args:
        conn (sqlite3.Connection): Open database connection.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            title      TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role            TEXT NOT NULL,
            content         TEXT NOT NULL,
            sources         TEXT,
            created_at      TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chat_messages_conv "
        "ON chat_messages(conversation_id, id)"
    )
    conn.commit()


def create_conversation(db_path: str, title: str = "Nouvelle conversation") -> int:
    """Create an empty conversation and return its id.

    Args:
        db_path (str): SQLite database path.
        title (str): Initial title (refined from the first question later).

    Returns:
        int: The new conversation's id.
    """
    conn = _connect(db_path)
    try:
        now = _now()
        cur = conn.execute(
            "INSERT INTO conversations (title, created_at, updated_at) "
            "VALUES (?, ?, ?)",
            (title[:120], now, now),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def add_message(db_path: str, conversation_id: int, role: str,
                content: str, sources: list[str] | None = None) -> None:
    """Append one message to a conversation and bump its updated_at.

    Args:
        db_path (str): SQLite database path.
        conversation_id (int): Target conversation.
        role (str): "user" or "assistant".
        content (str): Message text.
        sources (list[str] | None): Citation strings for an assistant message.
    """
    conn = _connect(db_path)
    try:
        now = _now()
        conn.execute(
            "INSERT INTO chat_messages "
            "(conversation_id, role, content, sources, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (conversation_id, role, content,
             json.dumps(sources or []), now),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )
        conn.commit()
    finally:
        conn.close()


def set_title(db_path: str, conversation_id: int, title: str) -> None:
    """Set a conversation's title (e.g. from its first user message).

    Args:
        db_path (str): SQLite database path.
        conversation_id (int): Target conversation.
        title (str): New title (truncated to a sane length).
    """
    conn = _connect(db_path)
    try:
        conn.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            (title[:120], conversation_id),
        )
        conn.commit()
    finally:
        conn.close()


def list_conversations(db_path: str) -> list[dict]:
    """Return all conversations, most recently updated first.

    Args:
        db_path (str): SQLite database path.

    Returns:
        list[dict]: One dict per conversation (id, title, updated_at).
    """
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations "
            "ORDER BY updated_at DESC"
        )
        return [dict(r) for r in cur]
    finally:
        conn.close()


def load_messages(db_path: str, conversation_id: int) -> list[dict]:
    """Return a conversation's messages in order, with decoded sources.

    Args:
        db_path (str): SQLite database path.
        conversation_id (int): Conversation to load.

    Returns:
        list[dict]: Each {"role", "content", "sources"} in chronological order.
    """
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "SELECT role, content, sources FROM chat_messages "
            "WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        )
        out = []
        for r in cur:
            try:
                sources = json.loads(r["sources"]) if r["sources"] else []
            except (ValueError, TypeError):
                sources = []
            out.append({"role": r["role"], "content": r["content"],
                        "sources": sources})
        return out
    finally:
        conn.close()


def delete_conversation(db_path: str, conversation_id: int) -> None:
    """Delete a conversation and all its messages.

    Args:
        db_path (str): SQLite database path.
        conversation_id (int): Conversation to delete.
    """
    conn = _connect(db_path)
    try:
        conn.execute(
            "DELETE FROM chat_messages WHERE conversation_id = ?",
            (conversation_id,),
        )
        conn.execute(
            "DELETE FROM conversations WHERE id = ?", (conversation_id,)
        )
        conn.commit()
    finally:
        conn.close()


def is_empty(db_path: str, conversation_id: int) -> bool:
    """Whether a conversation has no messages yet.

    Args:
        db_path (str): SQLite database path.
        conversation_id (int): Conversation to check.

    Returns:
        bool: True if the conversation has zero messages.
    """
    conn = _connect(db_path)
    try:
        n = conn.execute(
            "SELECT COUNT(*) FROM chat_messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()[0]
        return n == 0
    finally:
        conn.close()
