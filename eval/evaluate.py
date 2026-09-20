"""Evaluation harness for the PDF RAG pipeline.

Runs an end-to-end evaluation over a hand-written eval set (JSONL) and reports
two families of metrics:

  Retrieval quality (deterministic, no LLM cost)
    - hit@k          fraction of questions where at least one retrieved chunk
                     is judged relevant
    - recall@k       fraction of relevant chunks found (per question, averaged)
    - MRR            mean reciprocal rank of the first relevant chunk

  Answer quality (LLM-as-judge, via the project's active LLM provider)
    - faithfulness   is the answer grounded in the retrieved context? (1-5)
    - relevance      does it address the question? (1-5)
    - correctness    does it match the reference answer? (1-5)

A chunk counts as "relevant" for a question when it satisfies the eval item's
relevance signal:
    - expected_source     : chunk's filename must equal this, AND/OR
    - expected_substrings : chunk text must contain ALL these substrings
                            (case-insensitive)
If an item provides neither signal, retrieval metrics are skipped for it
(answer-quality is still graded).

Eval set format (one JSON object per line, see eval/eval_set.jsonl):
    {
      "question": "...",
      "reference_answer": "...",          # optional but needed for correctness
      "expected_source": "ocr.pdf",       # optional
      "expected_substrings": ["US01"]     # optional, list of strings
    }

Usage:
    set GROQ_API_KEY_1=your-key-here   (see .env.example for all key names)
    python eval/evaluate.py --db pdf_data.db --eval-set eval/eval_set.jsonl
    python eval/evaluate.py --top-k 8 --mode hybrid --report eval/report.json
    python eval/evaluate.py --no-judge          # retrieval metrics only

Run from the project root so `import config` / `from rag import ...` resolve.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field

# Make the project root importable when run as `python eval/evaluate.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from rag.generator import ask
from rag.retriever import RetrievedChunk, Retriever

# Windows consoles default to cp1252, which cannot encode the non-ASCII
# characters that appear in answers and document text (see main.py).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    except (AttributeError, ValueError):
        pass


# ---------------------------------------------------------------------------
# Eval-set loading
# ---------------------------------------------------------------------------

@dataclass
class EvalItem:
    """One evaluation question and its ground-truth signals.

    Attributes:
        question (str): The question to ask the pipeline.
        reference_answer (str): Gold answer for correctness grading.
        expected_source (str | None): Filename a relevant chunk must come from.
        expected_substrings (list[str]): Substrings a relevant chunk must contain.
    """

    question: str
    reference_answer: str = ""
    expected_source: str | None = None
    expected_substrings: list[str] = field(default_factory=list)

    def has_relevance_signal(self) -> bool:
        """Whether this item can be scored for retrieval relevance.

        Returns:
            bool: True if an expected_source or expected_substrings is set.
        """
        return bool(self.expected_source or self.expected_substrings)


def load_eval_set(path: str) -> list[EvalItem]:
    """Parse a JSONL eval set into EvalItem records.

    Args:
        path (str): Path to the JSONL file.

    Returns:
        list[EvalItem]: Parsed items, skipping blank lines.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a line is not valid JSON or lacks a question.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Eval set not found: {path}")

    items: list[EvalItem] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({e})") from e
            question = obj.get("question")
            if not question:
                raise ValueError(f"{path}:{lineno}: missing 'question'")
            items.append(EvalItem(
                question=question,
                reference_answer=obj.get("reference_answer", ""),
                expected_source=obj.get("expected_source"),
                expected_substrings=list(obj.get("expected_substrings", [])),
            ))
    return items


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def _is_relevant(chunk: RetrievedChunk, item: EvalItem) -> bool:
    """Decide whether a retrieved chunk satisfies an item's relevance signal.

    Args:
        chunk (RetrievedChunk): A retrieved chunk.
        item (EvalItem): The eval item with expected signals.

    Returns:
        bool: True if the chunk matches the source and all substrings.
    """
    if item.expected_source:
        if (chunk.filename or "").lower() != item.expected_source.lower():
            return False
    if item.expected_substrings:
        text = (chunk.text_content or "").lower()
        if not all(sub.lower() in text for sub in item.expected_substrings):
            return False
    return True


@dataclass
class RetrievalScore:
    """Retrieval metrics for a single question.

    Attributes:
        hit (bool): At least one relevant chunk was retrieved.
        recall (float): Relevant chunks retrieved / relevant chunks retrieved-or-expected.
        first_relevant_rank (int | None): 1-based rank of first relevant chunk.
        reciprocal_rank (float): 1 / first_relevant_rank, or 0.0.
        n_relevant (int): Count of relevant chunks among those retrieved.
        n_retrieved (int): Total chunks retrieved.
    """

    hit: bool
    recall: float
    first_relevant_rank: int | None
    reciprocal_rank: float
    n_relevant: int
    n_retrieved: int


def score_retrieval(chunks: list[RetrievedChunk], item: EvalItem) -> RetrievalScore:
    """Compute hit / recall / reciprocal-rank for one question's retrieval.

    Recall here is "of the chunks we surfaced, what share are relevant" —
    a precision-flavoured recall, since we have no exhaustive ground-truth
    set of all relevant chunks in the corpus. It still tracks whether the
    retriever is pulling on-target material.

    Args:
        chunks (list[RetrievedChunk]): Retrieved chunks, best-first.
        item (EvalItem): Eval item with relevance signals.

    Returns:
        RetrievalScore: Per-question retrieval metrics.
    """
    relevant_flags = [_is_relevant(c, item) for c in chunks]
    n_relevant = sum(relevant_flags)
    n_retrieved = len(chunks)

    first_rank = None
    for i, flag in enumerate(relevant_flags, 1):
        if flag:
            first_rank = i
            break

    recall = n_relevant / n_retrieved if n_retrieved else 0.0
    return RetrievalScore(
        hit=first_rank is not None,
        recall=recall,
        first_relevant_rank=first_rank,
        reciprocal_rank=(1.0 / first_rank) if first_rank else 0.0,
        n_relevant=n_relevant,
        n_retrieved=n_retrieved,
    )


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator of a retrieval-augmented QA system. "
    "Given a question, the retrieved context, the system's answer, and a "
    "reference answer, grade the system answer on three axes from 1 to 5:\n"
    "  - faithfulness: every claim is supported by the CONTEXT (not outside "
    "knowledge). 5 = fully grounded, 1 = hallucinated.\n"
    "  - relevance: the answer addresses the QUESTION. 5 = directly answers, "
    "1 = off-topic.\n"
    "  - correctness: the answer agrees with the REFERENCE answer. 5 = matches, "
    "1 = contradicts or misses. If no reference is given, judge against the "
    "context instead.\n"
    "A correct refusal ('the documents do not contain ...') when the reference "
    "also indicates the info is absent should score high on all axes.\n"
    'Respond with ONLY a JSON object: '
    '{"faithfulness": int, "relevance": int, "correctness": int, '
    '"rationale": "one sentence"}'
)


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM response.

    Args:
        text (str): Raw model output, possibly with prose around the JSON.

    Returns:
        dict: Parsed object, or {} if none could be parsed.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


@dataclass
class JudgeScore:
    """LLM-judge grades for one answer.

    Attributes:
        faithfulness (int): Grounding in context, 1-5 (0 = ungraded).
        relevance (int): Addresses the question, 1-5.
        correctness (int): Agreement with reference, 1-5.
        rationale (str): Judge's one-line justification.
    """

    faithfulness: int = 0
    relevance: int = 0
    correctness: int = 0
    rationale: str = ""


def judge_answer(question: str, context: str, answer: str,
                 reference: str, model: str) -> JudgeScore:
    """Grade a generated answer with the LLM judge.

    Args:
        question (str): The original question.
        context (str): The retrieved context shown to the generator.
        answer (str): The system's generated answer.
        reference (str): The gold reference answer (may be empty).
        model (str): Gemini chat model id to use as judge.

    Returns:
        JudgeScore: Parsed grades; zeros if the judge call/parse failed.
    """
    from rag.llm import generate

    user_message = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context or '(no context retrieved)'}\n\n"
        f"SYSTEM ANSWER:\n{answer}\n\n"
        f"REFERENCE ANSWER:\n{reference or '(none provided)'}\n\n"
        "Respond with ONLY a JSON object, no prose."
    )
    # Provider-agnostic + rotation-aware; _extract_json tolerates any wrapping.
    raw = generate(JUDGE_SYSTEM_PROMPT, user_message)
    data = _extract_json(raw or "")
    return JudgeScore(
        faithfulness=int(data.get("faithfulness", 0) or 0),
        relevance=int(data.get("relevance", 0) or 0),
        correctness=int(data.get("correctness", 0) or 0),
        rationale=str(data.get("rationale", "")),
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    """Arithmetic mean, returning 0.0 for an empty list.

    Args:
        values (list[float]): Numbers to average.

    Returns:
        float: The mean, or 0.0 if empty.
    """
    return sum(values) / len(values) if values else 0.0


def run_evaluation(args: argparse.Namespace) -> dict:
    """Execute the full evaluation and return a structured report.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        dict: Report with per-item results and aggregate metrics.
    """
    items = load_eval_set(args.eval_set)
    print(f"Loaded {len(items)} eval item(s) from {args.eval_set}")
    print(f"Retrieval: mode={args.mode} top_k={args.top_k} alpha={args.alpha} "
          f"rerank={args.rerank} query_transform={args.query_transform}")
    print(f"LLM judge: {'OFF' if args.no_judge else args.judge_model}\n")

    per_item: list[dict] = []

    with Retriever(args.db, model_name=args.model, default_mode=args.mode,
                   default_alpha=args.alpha, rerank=args.rerank,
                   query_transform=args.query_transform) as retriever:
        for idx, item in enumerate(items, 1):
            print(f"[{idx}/{len(items)}] {item.question}")

            if args.no_judge:
                # Retrieval-only run: skip generation so no answer tokens are
                # spent (expansion still runs if query_transform is on).
                chunks = retriever.retrieve(
                    item.question, top_k=args.top_k,
                    mode=args.mode, alpha=args.alpha,
                )
                answer = ""
            else:
                # Generation reuses the production ask() path so we evaluate
                # the exact pipeline users hit. stream=False keeps output clean.
                answer, chunks = ask(
                    retriever, item.question,
                    top_k=args.top_k, char_budget=args.char_budget,
                    mode=args.mode, alpha=args.alpha, stream=False,
                )

            result: dict = {
                "question": item.question,
                "answer": answer,
                "reference_answer": item.reference_answer,
                "retrieved": [
                    {"rank": r, "score": c.score, "citation": c.citation()}
                    for r, c in enumerate(chunks, 1)
                ],
            }

            if item.has_relevance_signal():
                rs = score_retrieval(chunks, item)
                result["retrieval"] = {
                    "hit": rs.hit,
                    "recall": round(rs.recall, 3),
                    "first_relevant_rank": rs.first_relevant_rank,
                    "reciprocal_rank": round(rs.reciprocal_rank, 3),
                    "n_relevant": rs.n_relevant,
                    "n_retrieved": rs.n_retrieved,
                }
                print(f"    retrieval: hit={rs.hit} recall={rs.recall:.2f} "
                      f"rr={rs.reciprocal_rank:.2f} "
                      f"(first relevant @ {rs.first_relevant_rank})")
            else:
                print("    retrieval: skipped (no expected_source/substrings)")

            if not args.no_judge:
                context = retriever.build_context(chunks, char_budget=args.char_budget)
                js = judge_answer(item.question, context, answer,
                                  item.reference_answer, args.judge_model)
                result["judge"] = {
                    "faithfulness": js.faithfulness,
                    "relevance": js.relevance,
                    "correctness": js.correctness,
                    "rationale": js.rationale,
                }
                print(f"    judge: faith={js.faithfulness} rel={js.relevance} "
                      f"corr={js.correctness}  {js.rationale}")

            per_item.append(result)
            print()

    # ---- Aggregates ----
    ret_items = [r["retrieval"] for r in per_item if "retrieval" in r]
    judge_items = [r["judge"] for r in per_item if "judge" in r]

    aggregate: dict = {"n_items": len(items)}
    if ret_items:
        aggregate["retrieval"] = {
            f"hit@{args.top_k}": round(_mean([1.0 if r["hit"] else 0.0 for r in ret_items]), 3),
            f"recall@{args.top_k}": round(_mean([r["recall"] for r in ret_items]), 3),
            "mrr": round(_mean([r["reciprocal_rank"] for r in ret_items]), 3),
            "n_scored": len(ret_items),
        }
    if judge_items:
        aggregate["answer_quality"] = {
            "faithfulness": round(_mean([j["faithfulness"] for j in judge_items]), 2),
            "relevance": round(_mean([j["relevance"] for j in judge_items]), 2),
            "correctness": round(_mean([j["correctness"] for j in judge_items]), 2),
            "n_scored": len(judge_items),
        }

    report = {
        "config": {
            "db": args.db,
            "mode": args.mode,
            "top_k": args.top_k,
            "alpha": args.alpha,
            "char_budget": args.char_budget,
            "rerank": args.rerank,
            "query_transform": args.query_transform,
            "judge_model": None if args.no_judge else args.judge_model,
        },
        "aggregate": aggregate,
        "items": per_item,
    }
    return report


def print_summary(report: dict) -> None:
    """Print the aggregate summary block to stdout.

    Args:
        report (dict): The report returned by run_evaluation.

    Returns:
        None
    """
    agg = report["aggregate"]
    print("=" * 60)
    print("AGGREGATE")
    print("=" * 60)
    print(f"items: {agg['n_items']}")
    if "retrieval" in agg:
        r = agg["retrieval"]
        print(f"\nRetrieval ({r['n_scored']} scored):")
        for k, v in r.items():
            if k != "n_scored":
                print(f"  {k:<14} {v}")
    if "answer_quality" in agg:
        q = agg["answer_quality"]
        print(f"\nAnswer quality ({q['n_scored']} scored, scale 1-5):")
        for k in ("faithfulness", "relevance", "correctness"):
            print(f"  {k:<14} {q[k]}")
    print("=" * 60)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the evaluator.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """
    p = argparse.ArgumentParser(description="Evaluate the PDF RAG pipeline.")
    p.add_argument("--db", default=config.DEFAULT_DB_PATH, help="SQLite database path")
    p.add_argument("--eval-set", default="eval/eval_set.jsonl",
                   help="Path to the JSONL eval set")
    p.add_argument("--mode", choices=["hybrid", "bm25", "semantic"], default="hybrid")
    p.add_argument("--alpha", type=float, default=config.DEFAULT_ALPHA)
    p.add_argument("--top-k", type=int, default=config.QA_TOP_K)
    p.add_argument("--char-budget", type=int, default=config.DEFAULT_CHAR_BUDGET)
    p.add_argument("--model", default=config.EMBEDDING_MODEL,
                   help="Sentence-transformer model for retrieval")
    p.add_argument("--judge-model", default="",
                   help="(Deprecated) the judge now uses the active LLM "
                        "provider chain; this flag is ignored.")
    p.add_argument("--no-judge", action="store_true",
                   help="Skip LLM grading; report retrieval metrics only")
    p.add_argument("--no-rerank", dest="rerank", action="store_false",
                   default=config.RERANK_ENABLED,
                   help="Disable cross-encoder reranking")
    p.add_argument("--query-transform", dest="query_transform",
                   action="store_true", default=config.QUERY_TRANSFORM_ENABLED,
                   help="Enable multi-query expansion")
    p.add_argument("--report", help="Write the full JSON report to this path")
    return p


def main() -> int:
    """Parse arguments, run the evaluation, print and optionally save results.

    Returns:
        int: Process exit code (0 on success, 1 on user/config error).
    """
    args = build_parser().parse_args()
    try:
        report = run_evaluation(args)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print_summary(report)

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nFull report written to {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
