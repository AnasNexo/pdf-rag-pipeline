"""Interactive chat loop with meta-query routing.

Wraps rag.generator in a REPL that additionally handles questions about
the database itself (document counts, listings, language detection) and
"summarise <doc>" commands directly from SQLite — no retrieval or LLM
call needed for those.
"""

from __future__ import annotations

import re

from config import QUERY_ROUTING_RULES
from rag.generator import ask, summarise
from rag.memory import ConversationMemory
from rag.query_transform import rewrite_with_history
from rag.retriever import RetrievedChunk, Retriever

# ---------------------------------------------------------------------------
# Meta-query patterns
# ---------------------------------------------------------------------------

# "How many documents / list documents" — about the DB, not document content.
_LIST_DOCS_RE = re.compile(
    r"\b(how many (docs?|documents?|pdfs?|files?)|list (the )?(docs?|documents?|pdfs?|files?)|"
    r"what (docs?|documents?|pdfs?|files?) (are|do you have|in (the )?db|in (the )?database)|"
    r"(docs?|documents?|pdfs?|files?) (do you have|in (the )?db|in (the )?database))\b",
    re.IGNORECASE,
)
# Qualifiers that indicate the question is about document content instead.
_LIST_DOCS_QUALIFIERS = re.compile(
    r"\b(contain|have|mention|include|about|with)\b",
    re.IGNORECASE,
)

# Language-detection query: "which documents are in French", "what language is X".
_LANG_QUERY_RE = re.compile(
    r"\b(which|what|are (any|the)).*\b(docs?|documents?|pdfs?|files?|one|ones)\b"
    r".*\b(language|french|english|german|spanish|arabic|chinese|japanese|italian"
    r"|portuguese|dutch|russian|korean|turkish)\b"
    r"|\b(what language|in what language|written in)\b",
    re.IGNORECASE,
)

# Language fingerprints — sampled high-frequency words per language.
_LANG_FINGERPRINTS: list[tuple[str, re.Pattern]] = [
    ("French",  re.compile(r"\b(en tant que|je veux|afin de|livreur|livraison|commande|parcourir|panier|client|produits|haute|basse|priorit[eé]|qualit[eé]|souhaite|disponibles|laisser|ameliorer|necesaire|necesit|avec|pour|dans|est|sont|les|des|une|qui|que|sur|par)\b", re.IGNORECASE)),
    ("English", re.compile(r"\b(the|and|for|that|this|with|have|from|are|will|been|their|were|which|they|more|also|than|when|would|about|there|could|other)\b", re.IGNORECASE)),
    ("German",  re.compile(r"\b(und|die|der|das|ist|von|mit|dem|für|sich|auf|den|eine|nicht|auch|werden|oder|über|bei|aus|nach)\b", re.IGNORECASE)),
    ("Spanish", re.compile(r"\b(el|la|los|las|de|que|en|un|una|por|con|más|pero|como|del|este|esta|para|sobre|entre|también)\b", re.IGNORECASE)),
]

# "What is rep.pdf about?" / "summarise test.pdf".
_DOC_ABOUT_RE = re.compile(
    r"\b(what is|what'?s|tell me about|summaris[ez]|describe|explain|overview of|about)\b.{0,60}?"
    r"([\w\-]+\.pdf)",
    re.IGNORECASE,
)

# "Summarise document 1" / "what is doc 2 about".
_SUMMARISE_DOC_RE = re.compile(
    r"\b(summaris[ez]|summarize|give me a summary of|what is|what'?s|tell me about|overview of)\b"
    r".{0,40}?\b(doc(?:ument)?)\s*(\d+|[\w\-]+\.pdf)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Meta-query handling
# ---------------------------------------------------------------------------

def _detect_doc_language(retriever: Retriever, doc_id: int) -> str:
    """Guess a document's language from a text sample.

    Samples up to 2000 characters of the document's first chunks and
    counts matches against per-language word fingerprints.

    Args:
        retriever (Retriever): Open retriever instance.
        doc_id (int): Document to inspect.

    Returns:
        str: Best-matching language name, or "Unknown".
    """
    rows = retriever.conn.execute(
        "SELECT text_content FROM chunks WHERE document_id = ? ORDER BY id LIMIT 5",
        (doc_id,),
    ).fetchall()
    sample = " ".join(r["text_content"] for r in rows)[:2000]
    best_lang = "Unknown"
    best_count = 0
    for lang_name, pattern in _LANG_FINGERPRINTS:
        count = len(pattern.findall(sample))
        if count > best_count:
            best_count = count
            best_lang = lang_name
    return best_lang


def handle_meta_query(
    user_input: str,
    retriever: Retriever,
    char_budget: int,
    stream: bool = True,
) -> bool:
    """Answer DB-level questions directly, bypassing retrieval and the LLM.

    Handles document counts/listings, language detection, and
    "summarise <doc>" phrasings.

    Args:
        user_input (str): Raw user message.
        retriever (Retriever): Open retriever instance.
        char_budget (int): Context budget forwarded to summarisation.
        stream (bool): Stream summarisation output to stdout.

    Returns:
        bool: True if the query was handled here; False if the caller
            should fall through to normal Q&A.
    """
    docs = retriever.list_documents()
    filename_map = {d["filename"].lower(): d for d in docs}
    id_map = {d["id"]: d for d in docs}

    # 1a. Language query: "which documents are in French?"
    if _LANG_QUERY_RE.search(user_input):
        lower_q = user_input.lower()
        target_lang: str | None = None
        for lang_name, _ in _LANG_FINGERPRINTS:
            if lang_name.lower() in lower_q:
                target_lang = lang_name
                break
        doc_langs = [(d, _detect_doc_language(retriever, d["id"])) for d in docs]
        if target_lang:
            matches = [d for d, lang in doc_langs if lang == target_lang]
            if matches:
                names = ", ".join(d["filename"] for d in matches)
                print(f"\nAssistant: The following document(s) appear to be "
                      f"in {target_lang}: {names}\n")
            else:
                print(f"\nAssistant: None of the documents appear to be "
                      f"in {target_lang}.\n")
        else:
            print("\nAssistant: Detected languages in the database:\n")
            for d, lang in doc_langs:
                print(f"  - {d['filename']} -> {lang}")
            print()
        return True

    # 1b. "How many documents / list documents" — skipped when the question
    # has content qualifiers like "which documents contain X". "do you have"
    # is removed first: its "have" is part of the listing phrasing itself,
    # not a content qualifier, and would otherwise veto the match.
    qualifier_text = re.sub(r"\bdo you have\b", "", user_input, flags=re.IGNORECASE)
    if _LIST_DOCS_RE.search(user_input) and not _LIST_DOCS_QUALIFIERS.search(qualifier_text):
        print(f"\nAssistant: There are {len(docs)} document(s) in the database:\n")
        for d in docs:
            print(f"  • [{d['id']}] {d['filename']}")
        print()
        return True

    # 2. "What is rep.pdf about?" / "summarise test.pdf"
    m = _DOC_ABOUT_RE.search(user_input)
    if m:
        doc = filename_map.get(m.group(2).lower())
        if doc:
            print("\nAssistant: ", end="", flush=True)
            summarise(retriever, doc["id"], char_budget=char_budget, stream=stream)
            print()
            return True

    # 3. "Summarise document 1" / "what is doc 2 about"
    m = _SUMMARISE_DOC_RE.search(user_input)
    if m:
        ref = m.group(3)
        if ref.lstrip("-").isdigit():
            doc = id_map.get(int(ref))
        else:
            doc = filename_map.get(ref.lower())
        if doc:
            print("\nAssistant: ", end="", flush=True)
            summarise(retriever, doc["id"], char_budget=char_budget, stream=stream)
            print()
            return True

    return False


def _route_to_document(user_input: str, retriever: Retriever) -> int | None:
    """Pick a document filter for a query, if one clearly applies.

    A document is selected when its filename appears verbatim in the
    query, or when a corpus-specific routing rule from
    config.QUERY_ROUTING_RULES matches.

    Args:
        user_input (str): Raw user message.
        retriever (Retriever): Open retriever instance.

    Returns:
        int | None: Document id to restrict retrieval to, or None.
    """
    docs = retriever.list_documents()
    lower_input = user_input.lower()

    for d in docs:
        if d["filename"].lower() in lower_input:
            return d["id"]

    for pattern, target_filename in QUERY_ROUTING_RULES:
        if re.search(pattern, user_input, re.IGNORECASE):
            doc = next((d for d in docs
                        if d["filename"].lower() == target_filename.lower()), None)
            if doc:
                return doc["id"]

    return None


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

def print_sources(chunks: list[RetrievedChunk]) -> None:
    """Print a deduplicated source list for an answer.

    Args:
        chunks (list[RetrievedChunk]): Chunks the answer was based on.

    Returns:
        None
    """
    if not chunks:
        return
    print("\nSources:")
    seen: set[str] = set()
    for c in chunks:
        cit = c.citation()
        if cit not in seen:
            print(f"  • {cit}  [{c.chunk_type}]")
            seen.add(cit)


def chat_loop(retriever: Retriever, top_k: int, char_budget: int,
              mode: str, alpha: float) -> None:
    """Run the interactive Q&A REPL until the user quits.

    Supports meta-queries, explicit "summarise <id|filename> [section]"
    commands, automatic document routing, and grounded Q&A with source
    citations.

    Args:
        retriever (Retriever): Open retriever instance.
        top_k (int): Chunks retrieved per question.
        char_budget (int): Maximum characters of context per prompt.
        mode (str): Retrieval mode (bm25 / semantic / hybrid).
        alpha (float): Hybrid blend weight.

    Returns:
        None
    """
    memory = ConversationMemory()

    docs = retriever.list_documents()
    doc_list = ", ".join(f"[{d['id']}] {d['filename']}" for d in docs)
    print(f"Q&A ready  (mode={mode}, top_k={top_k}).")
    print(f"Documents: {doc_list}")
    print('Ask anything, or type "summarise <id or filename.pdf>" to summarise '
          'a document, "forget" to clear conversation memory, "quit" to exit.\n')

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        if user_input.lower() in ("forget", "reset", "clear"):
            memory.clear()
            print("(conversation memory cleared)\n")
            continue

        # Meta-queries: "how many docs", "what is rep.pdf about", ...
        if handle_meta_query(user_input, retriever, char_budget=char_budget,
                             stream=True):
            continue

        # Explicit summarise command: summarise <id_or_filename> [section]
        parts = user_input.split(None, 2)
        if parts[0].lower() in ("summarise", "summarize") and len(parts) > 1:
            ref = parts[1]
            section = parts[2] if len(parts) > 2 else None
            docs = retriever.list_documents()
            if ref.lstrip("-").isdigit():
                doc = next((d for d in docs if d["id"] == int(ref)), None)
            else:
                doc = next((d for d in docs
                            if d["filename"].lower() == ref.lower()), None)
            if doc:
                print("\nAssistant: ", end="", flush=True)
                summarise(retriever, doc["id"], section=section,
                          char_budget=char_budget, stream=True)
                print()
            else:
                known = ", ".join(f"{d['id']}={d['filename']}" for d in docs)
                print(f"Document '{ref}' not found. Known documents: {known}\n")
            continue

        doc_id_filter = _route_to_document(user_input, retriever)

        # Resolve follow-up references ("its revenue?", "who wrote it?") into a
        # standalone query so retrieval searches for what the user means, while
        # the LLM still sees the original wording plus the full history.
        history = memory.as_transcript()
        retrieval_query = (
            rewrite_with_history(user_input, history)
            if not memory.is_empty() else user_input
        )

        print("\nAssistant: ", end="", flush=True)
        answer, chunks = ask(
            retriever, user_input,
            top_k=top_k, char_budget=char_budget,
            mode=mode, alpha=alpha, stream=True,
            document_id=doc_id_filter,
            history=history, retrieval_query=retrieval_query,
        )
        print_sources(chunks)
        memory.add(user_input, answer)
        print()
