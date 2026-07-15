# PLN-RAG Accuracy Benchmark

This folder contains hand-written benchmark cases for testing proof safety,
query alignment, negation handling, sufficiency questions, factor questions,
and topic transfer.

Use `cases.json` as the source of truth. Each case has:

- `id`: stable case id.
- `topic`: short domain label.
- `paragraph`: source text to ingest after a clean reset.
- `queries`: three questions for that paragraph.
- `expected_status`: expected proof result category.
- `expected_answer`: concise expected behavior.
- `rationale`: why that answer is correct.

Suggested manual workflow:

1. Reset the knowledge base.
2. Ingest one paragraph.
3. Ask its three questions.
4. Compare the answer, proof status, and executed query against the expected fields.

Important: this benchmark is designed to reward caution. If a question asks for
something not explicitly proved by the paragraph or by safe rules extracted from
it, the correct result is usually `unknown`, not a guessed yes/no answer.

Run all cases with:

```powershell
python benchmark\run_benchmark.py
```

Run focused cases while developing with one or more `--case-id` arguments. The
report separates direct-query accuracy from `unanswered` routing and records
overclaims, invalid targets, no-query failures, validated-proof rate, and p50/p95
latency. These metrics are the acceptance signal; overall accuracy alone is not.
