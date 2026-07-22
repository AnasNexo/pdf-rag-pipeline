"""LLM generation: grounded Q&A and summarisation.

Takes retrieved context from rag.retriever and produces answers with the
active LLM provider (rag.llm, which rotates across the configured Groq and
Gemini keys on quota errors). System prompts enforce strict grounding: the
model must answer only from the provided excerpts and cite sources inline.
"""

from __future__ import annotations

import textwrap

from config import DEFAULT_ALPHA, DEFAULT_CHAR_BUDGET, QA_TOP_K
from rag.llm import generate, generate_stream
from rag.retriever import RetrievedChunk, Retriever

QA_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a precise assistant that answers questions strictly from the provided document excerpts.

    Rules:
    - Base every claim on the excerpts below. Do not use outside knowledge.
    - If the excerpts do not contain the specific information asked for, say exactly:
      "The documents do not contain information about [topic]." Do not describe what
      the excerpts DO contain as a substitute answer.
    - Image descriptions tell you what an image looks like, NOT the values it displays.
      If a question asks for a specific measurement, number, or technical spec (e.g.
      frequency response, impedance, chart values) and the excerpt is an image description
      rather than text with those values, say: "The documents do not contain [topic] as
      explicit text — only an image that has not been OCR'd."
    - Cite sources inline using the format (filename, p.N) or (filename, "Section Name").
    - Keep answers concise unless the user asks for detail.
    - For tables or lists in the excerpts, preserve their structure in your answer.
""")

SUMMARY_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a precise summarisation assistant.

    Rules:
    - Summarise only from the provided document excerpts. Do not add outside knowledge.
    - Structure the summary with clear headings where the source material has sections.
    - Preserve key numbers, dates, and named entities exactly as they appear.
    - End with a one-sentence conclusion.
""")


def call_llm(system_prompt: str, user_message: str, stream: bool = True) -> str:
    """Send one system+user exchange to the active LLM (provider-agnostic).

    Rotation across providers/keys on quota errors is handled by rag.llm.
    When streaming, tokens are printed to stdout (CLI behaviour) and the full
    text returned; a web UI should use call_llm_stream instead.

    Args:
        system_prompt (str): System instructions.
        user_message (str): User message (context + question).
        stream (bool): If True, print tokens as they arrive and return the
            accumulated text; otherwise wait for the full response.

    Returns:
        str: The model's complete response text.

    Raises:
        RuntimeError: If no LLM API key is configured, or all are exhausted.
    """
    if stream:
        response_text = ""
        for piece in generate_stream(system_prompt, user_message):
            print(piece, end="", flush=True)
            response_text += piece
        print()
        return response_text

    return generate(system_prompt, user_message)


def call_llm_stream(system_prompt: str, user_message: str):
    """Stream an LLM response as a generator of text pieces (provider-agnostic).

    Unlike call_llm (which prints to stdout), this yields each token chunk so
    a caller — e.g. a web UI — can render the response progressively. Provider
    rotation on quota errors is handled by rag.llm.generate_stream.

    Args:
        system_prompt (str): System instructions.
        user_message (str): User message (context + question).

    Yields:
        str: Successive non-empty text pieces of the response.

    Raises:
        RuntimeError: If no LLM API key is configured, or all are exhausted.
    """
    yield from generate_stream(system_prompt, user_message)


def retrieve_for_answer(
    retriever: Retriever,
    query: str,
    top_k: int = QA_TOP_K,
    char_budget: int = DEFAULT_CHAR_BUDGET,
    mode: str = "hybrid",
    alpha: float = DEFAULT_ALPHA,
    document_id: int | None = None,
    history: str = "",
    retrieval_query: str | None = None,
) -> tuple[list[RetrievedChunk], str]:
    """Retrieve chunks and build the Q&A prompt, without calling the LLM.

    Splitting retrieval from generation lets a streaming UI show sources
    immediately, then stream the answer separately via call_llm_stream.

    Args:
        retriever (Retriever): Open retriever instance.
        query (str): The user's question (shown to the LLM).
        top_k (int): Number of chunks to retrieve.
        char_budget (int): Maximum characters of context for the prompt.
        mode (str): Retrieval mode (bm25 / semantic / hybrid).
        alpha (float): Hybrid blend weight.
        document_id (int | None): Restrict retrieval to one document.
        history (str): Recent conversation transcript for the prompt.
        retrieval_query (str | None): Standalone query to search with when it
            differs from ``query``. Defaults to ``query``.

    Returns:
        tuple[list[RetrievedChunk], str]: (chunks, prompt_message). The
            message is empty when nothing was retrieved.
    """
    chunks = retriever.retrieve(retrieval_query or query, top_k=top_k,
                                mode=mode, alpha=alpha, document_id=document_id)
    if not chunks:
        return [], ""

    context = retriever.build_context(chunks, char_budget=char_budget)
    message = _format_qa_message(query, context, history=history)
    return chunks, message


def _format_qa_message(query: str, context: str, history: str = "") -> str:
    """Assemble the user message for a Q&A call.

    Args:
        query (str): The user's question.
        context (str): Retrieved document excerpts.
        history (str): Recent conversation transcript, or "" for none.

    Returns:
        str: Formatted message: optional history, then excerpts, then question.
    """
    history_block = (
        f"Conversation so far:\n\n{history}\n\n---\n\n" if history.strip() else ""
    )
    return (f"{history_block}Document excerpts:\n\n{context}\n\n"
            f"---\n\nQuestion: {query}")


def _format_summary_message(context: str, scope: str) -> str:
    """Assemble the user message for a summarisation call.

    Args:
        context (str): Retrieved document excerpts.
        scope (str): Human-readable description of what is being summarised.

    Returns:
        str: Formatted summarisation request.
    """
    return f"Document excerpts ({scope}):\n\n{context}\n\n---\n\nPlease summarise the above."


def ask(
    retriever: Retriever,
    query: str,
    top_k: int = QA_TOP_K,
    char_budget: int = DEFAULT_CHAR_BUDGET,
    mode: str = "hybrid",
    alpha: float = DEFAULT_ALPHA,
    stream: bool = True,
    document_id: int | None = None,
    history: str = "",
    retrieval_query: str | None = None,
) -> tuple[str, list[RetrievedChunk]]:
    """Answer a question using retrieved document context.

    Args:
        retriever (Retriever): Open retriever instance.
        query (str): The user's question (as the user phrased it; shown to
            the LLM as the question to answer).
        top_k (int): Number of chunks to retrieve.
        char_budget (int): Maximum characters of context for the prompt.
        mode (str): Retrieval mode (bm25 / semantic / hybrid).
        alpha (float): Hybrid blend weight.
        stream (bool): Stream tokens to stdout while generating.
        document_id (int | None): Restrict retrieval to one document.
        history (str): Recent conversation transcript injected into the
            prompt so the answer can reference earlier turns. "" for none.
        retrieval_query (str | None): Standalone query to search with when it
            differs from ``query`` (e.g. a follow-up rewritten using history).
            Defaults to ``query``.

    Returns:
        tuple[str, list[RetrievedChunk]]: (answer_text, chunks_used).
            If nothing is retrieved, a fixed message and an empty list.
    """
    chunks = retriever.retrieve(retrieval_query or query, top_k=top_k,
                                mode=mode, alpha=alpha, document_id=document_id)
    if not chunks:
        return "No relevant content found in the documents.", []

    context = retriever.build_context(chunks, char_budget=char_budget)
    message = _format_qa_message(query, context, history=history)
    answer = call_llm(QA_SYSTEM_PROMPT, message, stream=stream)
    return answer, chunks


def summarise(
    retriever: Retriever,
    document_id: int,
    section: str | None = None,
    char_budget: int = DEFAULT_CHAR_BUDGET,
    stream: bool = True,
) -> str:
    """Summarise a whole document or a single section.

    Args:
        retriever (Retriever): Open retriever instance.
        document_id (int): Document to summarise.
        section (str | None): Restrict to one section title.
        char_budget (int): Maximum characters of context for the prompt.
        stream (bool): Stream tokens to stdout while generating.

    Returns:
        str: The summary text, or a "No content found" message.
    """
    chunks = retriever.retrieve_for_summary(document_id, section=section)
    scope = f"document {document_id}"
    if section:
        scope += f', section "{section}"'
    if not chunks:
        return f"No content found for {scope}."

    context = retriever.build_context(chunks, char_budget=char_budget)
    message = _format_summary_message(context, scope)
    return call_llm(SUMMARY_SYSTEM_PROMPT, message, stream=stream)
