"""Streamlit chat interface for the PDF RAG pipeline.

A conversational UI over the existing pipeline:

  - intuitive chat (st.chat_message / st.chat_input), answers streamed live;
  - conversation history kept across turns (st.session_state +
    rag.memory.ConversationMemory), with follow-up questions rewritten into
    standalone queries before retrieval;
  - sources cited under each answer (filename + page) via the chunk's own
    citation(), shown in an expander.

The sidebar exposes the pipeline's retrieval knobs (mode, query transform,
top_k) so the depth of the system is visible in a demo.

Run:
    set GROQ_API_KEY_1=your-key-here   (see .env.example for all key names)
    streamlit run app.py
"""

from __future__ import annotations

import os
import re
import tempfile

import streamlit as st

import config
from rag.generator import (
    QA_SYSTEM_PROMPT,
    _format_qa_message,
    call_llm_stream,
    retrieve_for_answer,
)
from rag import history
from rag.chunker import ChunkConfig
from rag.manage import add_pdfs, list_documents, rechunk_corpus, reset_corpus
from rag.memory import ConversationMemory
from rag.query_transform import rewrite_with_history
from rag.retriever import Retriever

st.set_page_config(page_title="RAG — Interrogation de documents", page_icon="📄",
                   layout="centered")


@st.cache_resource(show_spinner="Chargement du modèle et de la base…")
def get_retriever(db_path: str, rerank: bool) -> Retriever:
    """Build a Retriever once and cache it across reruns.

    Streamlit reruns the whole script on every interaction; caching the
    retriever keeps the embedding model and DB connection loaded instead of
    rebuilding them each message.

    Args:
        db_path (str): SQLite database path.
        rerank (bool): Whether the retriever reranks by default.

    Returns:
        Retriever: A ready, cached retriever.
    """
    # same_thread=False: Streamlit reruns may run on different threads, but
    # this cached connection is created once. Reads only, serialised — safe.
    return Retriever(db_path, rerank=rerank, same_thread=False)


def init_state() -> None:
    """Initialise session state for messages, memory, and the conversation id.

    On first load (or after the conversation was deleted), a fresh empty
    conversation is created in the database so every message has a home.
    """
    if "messages" not in st.session_state:
        # Each message: {"role", "content", "sources": list[str] | None}.
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory()
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = history.create_conversation(
            config.DEFAULT_DB_PATH)


def load_conversation(conversation_id: int) -> None:
    """Load a saved conversation into the live chat state.

    Restores both the rendered messages and the ConversationMemory (so
    follow-up questions still resolve against the reloaded history).

    Args:
        conversation_id (int): Conversation to load.
    """
    msgs = history.load_messages(config.DEFAULT_DB_PATH, conversation_id)
    st.session_state.messages = msgs
    mem = ConversationMemory()
    # Rebuild memory from consecutive user/assistant pairs.
    pending_q = None
    for m in msgs:
        if m["role"] == "user":
            pending_q = m["content"]
        elif m["role"] == "assistant" and pending_q is not None:
            mem.add(pending_q, m["content"])
            pending_q = None
    st.session_state.memory = mem
    st.session_state.conversation_id = conversation_id


def new_conversation() -> None:
    """Start a fresh, empty conversation."""
    st.session_state.messages = []
    st.session_state.memory = ConversationMemory()
    st.session_state.conversation_id = history.create_conversation(
        config.DEFAULT_DB_PATH)


def persist_turn(user_msg: str, assistant_msg: str,
                 sources: list[str]) -> None:
    """Save a completed user/assistant turn to the database.

    Titles a brand-new conversation from its first user message.

    Args:
        user_msg (str): The user's message.
        assistant_msg (str): The assistant's reply.
        sources (list[str]): Citations for the assistant message.
    """
    cid = st.session_state.conversation_id
    was_empty = history.is_empty(config.DEFAULT_DB_PATH, cid)
    history.add_message(config.DEFAULT_DB_PATH, cid, "user", user_msg)
    history.add_message(config.DEFAULT_DB_PATH, cid, "assistant",
                        assistant_msg, sources)
    if was_empty:
        history.set_title(config.DEFAULT_DB_PATH, cid, user_msg)


# Meta-queries about the corpus itself ("how many docs", "list documents"),
# as opposed to questions about document *content*. Answered directly from the
# document list — no retrieval, no LLM call.
_META_LIST_RE = re.compile(
    r"\b(how many|combien)\b.*\b(doc|document|fichier|pdf|file)s?\b"
    r"|\b(list|liste|quels?|which|what)\b.*\b(doc|document|fichier|pdf|file)s?\b"
    r"|\b(doc|document|fichier|pdf|file)s?\b.*\b(do you have|y a-t-il|disponibles?)\b",
    re.IGNORECASE,
)
# "about/contain/mention" make it a content question — let it fall through.
_META_CONTENT_RE = re.compile(
    r"\b(about|contain|mention|sur|parle|contient|à propos)\b", re.IGNORECASE,
)

# A filename mentioned anywhere in the message (e.g. "summarise ocr.pdf",
# "what is im.pdf about?"). Document-level questions must retrieve FROM that
# document by id, not text-search for the filename string (which isn't in the
# chunk text) — otherwise the LLM wrongly says the file isn't in the corpus.
_FILENAME_RE = re.compile(r"\b([\w\-]+\.pdf)\b", re.IGNORECASE)


def _match_document(name: str, docs: list[dict]) -> dict | None:
    """Match a (possibly mistyped or prefixed) filename against the documents.

    Tries, in order:
      1. exact filename match;
      2. stem prefix match in either direction ("img" -> "im.pdf");
      3. stem containment ("im" -> "tmp1ptuzvg6_im.pdf"), which also rescues
         documents stored with a temp-upload prefix.
    Returns the single best match, or None if zero or ambiguous (>1) match —
    ambiguity is safer to refuse than to guess wrong.

    Args:
        name (str): The filename the user typed (e.g. "im.pdf").
        docs (list[dict]): Documents, each with a "filename" and "id".

    Returns:
        dict | None: The matched document dict, or None.
    """
    name = name.lower()
    for d in docs:
        if d["filename"].lower() == name:
            return d  # exact match wins outright

    stem = name[:-4] if name.endswith(".pdf") else name
    if not stem:
        return None

    # Prefix match (either direction) first; fall back to containment.
    for test in (
        lambda a, b: a.startswith(b) or b.startswith(a),
        lambda a, b: a in b or b in a,
    ):
        candidates = [d for d in docs
                      if test(d["filename"].lower()[:-4], stem)]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            return None  # ambiguous — don't guess
    return None


# "for each/all documents …", "describe every file", "tous les documents …".
# These need a chunk from EVERY document, not a single top-k retrieval (which
# only covers whatever ranks highest and silently omits some documents).
_ALL_DOCS_RE = re.compile(
    r"\b(each|every|all|chaque|tous|toutes)\b.{0,30}"
    r"\b(doc|document|fichier|pdf|file)s?\b"
    r"|\b(doc|document|fichier|pdf|file)s?\b.{0,30}\b(each|every|all|chacun)\b",
    re.IGNORECASE,
)


def build_all_documents_context(retriever: Retriever,
                                per_doc: int = 3) -> tuple[str, list[str]]:
    """Assemble context with representative chunks from EVERY document.

    For corpus-wide questions ("introduce each document"), a single ranked
    retrieval misses whole documents. This instead pulls the first few chunks
    of each document in turn, so the LLM sees all of them.

    Args:
        retriever (Retriever): Open retriever.
        per_doc (int): Chunks to include per document.

    Returns:
        tuple[str, list[str]]: (context_text, source_citations).
    """
    parts: list[str] = []
    sources: list[str] = []
    for d in retriever.list_documents():
        chunks = retriever.retrieve_for_summary(d["id"])[:per_doc]
        if not chunks:
            continue
        body = "\n".join(c.text_content for c in chunks)
        parts.append(f"=== Document: {d['filename']} ===\n{body}")
        sources.append(d["filename"])
    return "\n\n".join(parts), sources


def detect_document_scope(prompt: str,
                          retriever: Retriever) -> tuple[int | None, str | None]:
    """Resolve a filename in the question to a document id.

    Args:
        prompt (str): The user's message.
        retriever (Retriever): Open retriever (for the document list).

    Returns:
        tuple[int | None, str | None]: ``(doc_id, unknown_name)``.
            - ``(id, None)``   — a document was named and matched;
            - ``(None, name)`` — a .pdf was named but matches no document;
            - ``(None, None)`` — no filename was mentioned at all.
    """
    m = _FILENAME_RE.search(prompt)
    if not m:
        return None, None
    name = m.group(1)
    doc = _match_document(name, retriever.list_documents())
    if doc:
        return doc["id"], None
    return None, name


def handle_meta_query(prompt: str, retriever: Retriever) -> str | None:
    """Answer corpus-level questions directly, without retrieval.

    Args:
        prompt (str): The user's message.
        retriever (Retriever): Open retriever (for the document list).

    Returns:
        str | None: A direct answer, or None to fall through to normal Q&A.
    """
    if not _META_LIST_RE.search(prompt) or _META_CONTENT_RE.search(prompt):
        return None
    docs = retriever.list_documents()
    if not docs:
        return "Aucun document n'est chargé dans la base."
    lines = [f"Il y a **{len(docs)} document(s)** dans la base :", ""]
    lines += [f"- `{d['filename']}`" for d in docs]
    return "\n".join(lines)


def render_sources(sources: list[str]) -> None:
    """Render a citation list under an answer.

    Args:
        sources (list[str]): Pre-formatted citation strings (filename + page).
    """
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for src in sources:
            st.markdown(f"- {src}")


# ---------------------------------------------------------------------------
# Sidebar — pipeline controls
# ---------------------------------------------------------------------------

init_state()

with st.sidebar:
    # --- Conversation history (ChatGPT-style) ---
    st.header("💬 Conversations")
    if st.button("➕ Nouvelle conversation", use_container_width=True):
        new_conversation()
        st.rerun()

    convos = history.list_conversations(config.DEFAULT_DB_PATH)
    current_id = st.session_state.conversation_id
    for c in convos:
        is_current = c["id"] == current_id
        col_open, col_del = st.columns([5, 1])
        label = ("▸ " if is_current else "") + (c["title"] or "Sans titre")
        if col_open.button(label, key=f"open_{c['id']}",
                           use_container_width=True,
                           disabled=is_current):
            load_conversation(c["id"])
            st.rerun()
        if col_del.button("🗑️", key=f"del_{c['id']}",
                          use_container_width=True):
            history.delete_conversation(config.DEFAULT_DB_PATH, c["id"])
            # If we deleted the open one, fall back to a fresh conversation.
            if is_current:
                new_conversation()
            st.rerun()

    st.divider()
    with st.expander("⚙️ Paramètres", expanded=False):
        mode = st.selectbox(
            "Mode de recherche", ["hybrid", "semantic", "bm25"],
            index=0,
            help="hybrid combine BM25 (mots-clés) et recherche sémantique.",
        )
        query_transform = st.toggle(
            "Expansion de requête", value=config.QUERY_TRANSFORM_ENABLED,
            help="Reformule la question en plusieurs variantes (meilleur "
                 "rappel, mais un appel LLM de plus par question).",
        )
        top_k = st.slider("Nombre de passages (top_k)", min_value=3,
                          max_value=15, value=config.QA_TOP_K)

    with st.expander("📁 Ajouter des PDF", expanded=False):
        # --- Upload & process new PDFs ---
        uploads = st.file_uploader(
            "Fichiers PDF", type=["pdf"], accept_multiple_files=True,
            help="Sélectionnez un ou plusieurs PDF, puis lancez le traitement.",
        )

        # --- Chunking method (chosen at ingestion time) ---
        chunk_method = st.radio(
            "Méthode de découpage",
            ["Par nombre de mots", "Sémantique (par thème)"],
            index=1 if config.SEMANTIC_CHUNKING_ENABLED else 0,
            help="« Par nombre de mots » découpe tous les ~300 mots. "
                 "« Sémantique » découpe là où le sujet change (embeddings de "
                 "phrases) : passages plus cohérents, ingestion un peu plus lente.",
        )
        use_semantic = chunk_method.startswith("Sémantique")

        if st.button("⚙️ Traiter les PDF", use_container_width=True,
                     disabled=not uploads):
            # Streamlit gives in-memory files; the pipeline needs file paths
            # and stores the path's basename as the document name. Write each
            # upload into a temp DIRECTORY under its REAL name so the stored
            # filename is clean (not "tmpXXXX_name.pdf").
            #
            # NOT TemporaryDirectory(): on Windows the PDF libraries may still
            # hold a handle when the context exits, and its strict auto-cleanup
            # then raises PermissionError. Create the dir manually and remove
            # it with ignore_errors=True (the OS reclaims temp files later).
            import shutil
            tmp_dir = tempfile.mkdtemp(prefix="ragupload_")
            try:
                tmp_paths = []
                for up in uploads:
                    safe = os.path.basename(up.name)  # strip path components
                    path = os.path.join(tmp_dir, safe)
                    with open(path, "wb") as f:
                        f.write(up.getbuffer())
                    tmp_paths.append(path)
                with st.spinner("Traitement (ingestion → chunking → "
                                "indexation)…"):
                    result = add_pdfs(
                        tmp_paths, config.DEFAULT_DB_PATH,
                        chunk_config=ChunkConfig(semantic=use_semantic),
                    )
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            if result.ingested:
                method_label = ("sémantique" if use_semantic
                                else "par nombre de mots")
                st.success(f"{len(result.ingested)} PDF traité(s) · "
                           f"{result.chunks_added} chunks ({method_label}) · "
                           f"{result.embedded} embeddings.")
            if result.skipped:
                st.info("Déjà présent(s), ignoré(s) : "
                        + ", ".join(f"`{n}`" for n in result.skipped))
            for name, err in result.failed:
                st.error(f"Échec : {name} — {err}")
            # New data on disk: drop the cached retriever so it reopens fresh.
            get_retriever.clear()
            st.rerun()

        # --- Current corpus ---
        docs = list_documents(config.DEFAULT_DB_PATH)
        if docs:
            if len(docs) <= 20:  # only list when the count is manageable
                st.caption(f"{len(docs)} document(s) :")
                for d in docs:
                    st.markdown(f"- `{d['filename']}`")
            else:
                st.caption(f"{len(docs)} documents dans la base.")
        else:
            st.caption("Aucun document. Ajoutez des PDF ci-dessus.")

        # --- Re-chunk existing corpus with the selected method (no re-upload) ---
        # Switching chunking method only needs re-chunk + re-index, not a full
        # re-parse. This button applies the "Méthode de découpage" choice above
        # to the documents already in the base.
        if docs:
            if st.button("🔄 Re-découper le corpus avec cette méthode",
                         use_container_width=True,
                         help="Reconstruit les chunks des documents déjà "
                              "présents avec la méthode choisie ci-dessus. "
                              "Pas besoin de re-téléverser les PDF."):
                with st.spinner("Re-découpage et ré-indexation…"):
                    n_chunks, n_emb = rechunk_corpus(
                        config.DEFAULT_DB_PATH, semantic=use_semantic)
                get_retriever.clear()  # chunks changed: reopen retriever fresh
                lbl = "sémantique" if use_semantic else "par nombre de mots"
                st.success(f"Corpus re-découpé ({lbl}) : "
                           f"{n_chunks} chunks · {n_emb} embeddings.")
                st.rerun()

        # --- Delete everything (two-click confirm) ---
        if docs:
            if st.session_state.get("confirm_reset"):
                st.warning("Supprimer **tous** les documents ?")
                c1, c2 = st.columns(2)
                if c1.button("Oui, supprimer", use_container_width=True):
                    n = reset_corpus(config.DEFAULT_DB_PATH)
                    get_retriever.clear()
                    st.session_state.confirm_reset = False
                    st.session_state.messages = []
                    st.session_state.memory.clear()
                    st.toast(f"{n} document(s) supprimé(s).")
                    st.rerun()
                if c2.button("Annuler", use_container_width=True):
                    st.session_state.confirm_reset = False
                    st.rerun()
            elif st.button("🗑️ Supprimer tous les documents",
                           use_container_width=True):
                st.session_state.confirm_reset = True
                st.rerun()


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

# Smaller than st.title so the heading fits on a single line.
st.markdown(
    "<h2 style='font-size:1.6rem; font-weight:700; white-space:nowrap; "
    "margin:0 0 0.5rem 0;'>Posez votre question, je m'occupe du reste.</h2>",
    unsafe_allow_html=True,
)

# No corpus yet (fresh install or after a full delete): prompt to upload and
# stop before building a retriever over a missing/empty database.
if not list_documents(config.DEFAULT_DB_PATH):
    st.info("Aucun document chargé. Utilisez **📁 Ajouter des PDF** dans la "
            "barre latérale pour ajouter des PDF.")
    st.stop()

retriever = get_retriever(config.DEFAULT_DB_PATH, config.RERANK_ENABLED)

# Replay the conversation so far.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources") or [])

# Handle a new question.
if prompt := st.chat_input("Votre question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    memory = st.session_state.memory
    with st.chat_message("assistant"):
        # Corpus-level questions ("how many documents?") are answered directly,
        # bypassing retrieval and the LLM (and the daily quota).
        meta_answer = handle_meta_query(prompt, retriever)
        if meta_answer is not None:
            st.markdown(meta_answer)
            answer, sources = meta_answer, []
            memory.add(prompt, answer)
            persist_turn(prompt, answer, sources)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
            st.stop()

        # "Introduce each document" / "describe all the files": build context
        # from every document so none is silently omitted, then answer.
        if _ALL_DOCS_RE.search(prompt):
            hist = memory.as_transcript()
            with st.spinner("Lecture de tous les documents…"):
                context, all_sources = build_all_documents_context(retriever)
            if not context:
                answer = "Aucun document n'est chargé dans la base."
                st.markdown(answer)
                sources = []
            else:
                message = _format_qa_message(prompt, context, history=hist)
                answer = st.write_stream(
                    call_llm_stream(QA_SYSTEM_PROMPT, message))
                sources = all_sources
                render_sources(sources)
            memory.add(prompt, answer)
            persist_turn(prompt, answer, sources)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
            st.stop()

        # If the question names a specific file ("summarise ocr.pdf",
        # "what is im.pdf about?"), scope retrieval to that document so we
        # pull ITS content instead of text-searching for the filename.
        doc_id, unknown_file = detect_document_scope(prompt, retriever)

        # A .pdf was named that doesn't exist (e.g. a typo "img.pdf"): say so
        # helpfully instead of falling through to a misleading "not found".
        if unknown_file is not None:
            available = ", ".join(f"`{d['filename']}`"
                                  for d in retriever.list_documents())
            answer = (f"Aucun fichier nommé `{unknown_file}` n'existe. "
                      f"Documents disponibles : {available}.")
            st.markdown(answer)
            memory.add(prompt, answer)
            persist_turn(prompt, answer, [])
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": []}
            )
            st.stop()

        # Resolve follow-up references ("its revenue?") into a standalone query
        # for retrieval, while the LLM still sees the original wording + history.
        hist = memory.as_transcript()
        if doc_id is not None:
            # The filename is a poor search query; broaden it so retrieval
            # surfaces representative chunks from the scoped document.
            retrieval_query = "résumé contenu principal sujet du document"
        else:
            retrieval_query = (
                rewrite_with_history(prompt, hist)
                if not memory.is_empty() else prompt
            )

        with st.spinner("Recherche dans les documents…"):
            chunks, message = retrieve_for_answer(
                retriever, prompt,
                top_k=top_k, mode=mode, document_id=doc_id,
                history=hist, retrieval_query=retrieval_query,
            )

        if not chunks:
            answer = "Aucun contenu pertinent trouvé dans les documents."
            st.markdown(answer)
            sources: list[str] = []
        else:
            # Stream the answer token-by-token.
            answer = st.write_stream(call_llm_stream(QA_SYSTEM_PROMPT, message))
            # Deduplicated, ordered citations (filename + page).
            seen: set[str] = set()
            sources = []
            for c in chunks:
                cit = c.citation()
                if cit not in seen:
                    seen.add(cit)
                    sources.append(cit)
            render_sources(sources)

        memory.add(prompt, answer)
        persist_turn(prompt, answer, sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
