"""Query transformation: multi-query expansion via the LLM.

A single question phrasing only matches chunks worded similarly. Multi-query
expansion asks the LLM to rewrite the question into several semantically
equivalent variants (synonyms, reordering, expanded acronyms, formal vs.
colloquial). First-stage search then runs for every variant and the candidate
pools are fused, so a chunk that any phrasing surfaces becomes a candidate —
widening recall before the cross-encoder reranks.

The expansion is a single non-streaming LLM call. It degrades gracefully: if
the call or parse fails, the original query is used alone, so retrieval never
breaks because expansion did.
"""

from __future__ import annotations

import re
import textwrap

from config import QUERY_EXPANSION_N

EXPANSION_SYSTEM_PROMPT = textwrap.dedent("""\
    You rewrite a search question into alternative phrasings to improve document
    retrieval. Produce semantically equivalent variants that a relevant passage
    might match: use synonyms, reorder clauses, expand or contract acronyms, and
    vary between formal and colloquial wording. Preserve the original meaning and
    every named entity; do not answer the question or add new facts.

    Respond with ONLY the variants, one per line, no numbering or bullets.
""")


def _parse_variants(text: str) -> list[str]:
    """Extract clean, deduplicated variant lines from an LLM response.

    Strips any leading numbering, bullets, or quotes the model may add.

    Args:
        text (str): Raw model output, one variant per line.

    Returns:
        list[str]: Cleaned variant strings, order preserved, blanks dropped.
    """
    variants: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        cleaned = re.sub(r'^\s*(?:[-*•]|\d+[.)])\s*', '', line).strip()
        cleaned = cleaned.strip('"\'')
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            variants.append(cleaned)
    return variants


def expand_query(
    query: str,
    n: int = QUERY_EXPANSION_N,
) -> list[str]:
    """Expand a query into the original plus up to *n* LLM-generated variants.

    The original query is always first in the returned list so retrieval still
    favours the user's exact phrasing. Falls back to ``[query]`` on any error.

    Args:
        query (str): The user's question.
        n (int): Number of additional variants to request.

    Returns:
        list[str]: ``[query, variant1, ...]`` with at most ``n + 1`` entries,
            deduplicated.
    """
    if n <= 0:
        return [query]

    # Imported here so retrieval has no hard dependency on the Gemini client
    # unless expansion is actually requested.
    from rag.generator import call_llm

    user_message = f"Question: {query}\n\nWrite {n} alternative phrasings."
    try:
        raw = call_llm(EXPANSION_SYSTEM_PROMPT, user_message, stream=False)
    except Exception as e:  # noqa: BLE001 — never let expansion break retrieval
        print(f"  Query expansion failed ({e}); using original query only.")
        return [query]

    variants = _parse_variants(raw)[:n]
    queries = [query]
    for v in variants:
        if v.lower() != query.lower():
            queries.append(v)
    return queries


REWRITE_SYSTEM_PROMPT = textwrap.dedent("""\
    You rewrite a follow-up question into a standalone search query using the
    conversation so far. Resolve pronouns and references ("it", "that company",
    "there") into the explicit entities they refer to, so the query makes sense
    on its own without the conversation. Keep it concise. If the question is
    already self-contained, return it unchanged. Output ONLY the rewritten
    query, nothing else.
""")


def rewrite_with_history(
    query: str,
    history_transcript: str,
) -> str:
    """Rewrite a follow-up question into a standalone query using history.

    Lets retrieval search for what the user actually means ("what was its
    revenue?" → "what was Microsoft's revenue?") instead of the literal,
    context-free wording. Falls back to the original query on any error or
    empty result, so a failed rewrite never blocks the turn.

    Args:
        query (str): The user's (possibly elliptical) follow-up.
        history_transcript (str): Recent turns as text (see
            ConversationMemory.as_transcript). Empty means no rewrite needed.

    Returns:
        str: The standalone query, or the original if unchanged/on failure.
    """
    if not history_transcript.strip():
        return query

    from rag.generator import call_llm

    user_message = (
        f"Conversation so far:\n{history_transcript}\n\n"
        f"Follow-up question: {query}\n\nStandalone query:"
    )
    try:
        rewritten = call_llm(REWRITE_SYSTEM_PROMPT, user_message, stream=False)
    except Exception as e:  # noqa: BLE001 — never let rewriting break the turn
        print(f"  Query rewrite failed ({e}); using original question.")
        return query

    rewritten = rewritten.strip().strip('"\'')
    return rewritten or query


def fuse_rankings(ranked_lists: list[list[dict]], rrf_k: int) -> list[dict]:
    """Fuse several ranked candidate lists into one via Reciprocal Rank Fusion.

    Each list is one query variant's first-stage results. A chunk's fused score
    sums ``1 / (rrf_k + rank)`` across every list it appears in, so chunks
    surfaced by multiple phrasings rise to the top. Mirrors the RRF used in
    rag.search.hybrid_search.

    Args:
        ranked_lists (list[list[dict]]): Per-variant result dicts, best-first.
            Each dict has at least ``chunk_id``.
        rrf_k (int): RRF constant (config.RRF_K).

    Returns:
        list[dict]: Unique candidate dicts ordered by fused score, with that
            score written to each dict's ``score`` field.
    """
    pool: dict[int, dict] = {}
    scores: dict[int, float] = {}
    for results in ranked_lists:
        for rank, r in enumerate(results, 1):
            cid = r["chunk_id"]
            pool.setdefault(cid, r)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)

    ranked = sorted(pool, key=lambda c: scores[c], reverse=True)
    return [{**pool[c], "score": scores[c]} for c in ranked]
