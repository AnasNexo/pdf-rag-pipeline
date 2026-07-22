# RAG pipeline evaluation

`evaluate.py` runs an end-to-end evaluation of the pipeline (retrieval +
generation) over a hand-written eval set and reports both **retrieval quality**
and **answer quality**.

## 1. Prerequisites

The corpus must already be ingested, chunked, and indexed (the `eval/` script
does not build anything — it queries the existing DB):

```bat
python main.py ingest data\rep.pdf data\ocr.pdf
python main.py chunk
python main.py index
```

For the LLM judge, set your key (same ones the pipeline uses; see
`.env.example` for the full list):

```bat
set GROQ_API_KEY_1=your-key-here
```

## 2. Write the eval set

Edit `eval/eval_set.jsonl`. One JSON object per line:

| field                 | required | meaning                                                        |
|-----------------------|----------|----------------------------------------------------------------|
| `question`            | yes      | the question to ask the pipeline                               |
| `reference_answer`    | no\*     | gold answer; used for the judge's **correctness** score        |
| `expected_source`     | no       | filename a *relevant* chunk must come from (e.g. `ocr.pdf`)    |
| `expected_substrings` | no       | list of strings a relevant chunk must contain (case-insensitive)|

\* Without a `reference_answer`, correctness is judged against the retrieved
context instead of a gold answer.

A chunk is counted **relevant** for retrieval metrics when it matches
`expected_source` **and** contains **all** `expected_substrings`. If an item has
neither, retrieval metrics are skipped for it (the answer is still judged).

Example:

```json
{"question": "Que décrit l'user story US01?", "reference_answer": "US01 décrit ...", "expected_source": "ocr.pdf", "expected_substrings": ["US01"]}
```

## 3. Run

```bat
:: full eval (retrieval metrics + LLM judge)
python eval\evaluate.py

:: retrieval metrics only — no API key, no cost, deterministic
python eval\evaluate.py --no-judge

:: try a different retrieval config and save a report
python eval\evaluate.py --mode semantic --top-k 6 --report eval\report.json
```

Run from the project root so `import config` and `from rag import ...` resolve.

## 4. What the metrics mean

**Retrieval** (deterministic):
- `hit@k` — share of questions with ≥1 relevant chunk in the top-k.
- `recall@k` — of the chunks surfaced, the share that are relevant (averaged).
- `mrr` — mean reciprocal rank of the first relevant chunk (1.0 = always rank 1).

**Answer quality** (LLM judge, 1–5):
- `faithfulness` — is every claim grounded in the retrieved context?
- `relevance` — does the answer address the question?
- `correctness` — does it match the reference answer?

A correct refusal ("The documents do not contain information about …") scores
high when the reference also indicates the info is absent.

## 5. Comparing configurations

To tune retrieval, sweep a parameter and compare reports:

```bat
python eval\evaluate.py --mode bm25     --report eval\bm25.json
python eval\evaluate.py --mode semantic --report eval\semantic.json
python eval\evaluate.py --alpha 0.3     --report eval\hybrid_a03.json
python eval\evaluate.py --alpha 0.8     --report eval\hybrid_a08.json
```

Each report's `aggregate` block holds the headline numbers to compare.
