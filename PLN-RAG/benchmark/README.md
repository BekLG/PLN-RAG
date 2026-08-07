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

## stress25_v1 (discovery suite, no labels)

`stress25_v1.json` holds 25 real-world cases: 15 paper abstracts, 7 web snippet
bundles, and 3 EntailmentBank items, each with a single `user_query`.

It carries **no `expected_status` labels**, so it cannot measure accuracy. Run it
to measure whether the query path reaches an executable target on real text, and
which gate discards candidates when it does not:

```powershell
python benchmark\run_stress_suite.py
python benchmark\run_stress_suite.py --case-id A01 --case-id A02
```

Reported: `executed_rate`, `no_query_rate`, `weakly_aligned_rate`,
`proof_found_rate`, `rejection_stages`, and `suppressed_candidate_count` — cases
where the parser produced a query that no execution candidate survived from. That
last number is the direct measure of over-aggressive candidate filtering.

Because `cases.json` is hand-written and its vocabulary is reflected in the query
gates, treat it as a regression guard and use this suite plus fresh probe cases to
measure progress.

Run focused cases while developing with one or more `--case-id` arguments. The
report separates direct-query accuracy from `unanswered` routing and records
overclaims, invalid targets, no-query failures, validated-proof rate, and p50/p95
latency. These metrics are the acceptance signal; overall accuracy alone is not.
