"""
Run the stress25_v1 suite against a live PLN-RAG API.

This suite carries no `expected_status` labels, so it CANNOT measure accuracy.
It measures whether the query path reaches an executable target at all, and which
gate discards candidates when it does not. Those are the signals for
`no_semantically_compatible_query` regressions on real-world text — the bundled
`cases.json` is hand-written and much easier.

Schema differs from `cases.json`, hence the adapter below:
    case_id    -> id
    input_text -> paragraph
    user_query -> a single question (no expected status)

Usage:
    python benchmark/run_stress_suite.py
    python benchmark/run_stress_suite.py --case-id A01 --case-id A02
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_benchmark import http_json, percentile

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmark"
DEFAULT_SUITE = BENCHMARK_DIR / "stress25_v1.json"
DEFAULT_RESULTS_DIR = BENCHMARK_DIR / "results" / "stress"

BANNER = """
+----------------------------------------------------------------------------+
| stress25_v1 has no expected_status labels.                                  |
| These numbers describe proof DISCOVERY and gate behavior, NOT correctness.  |
| Do not report them as accuracy.                                            |
+----------------------------------------------------------------------------+
"""


def load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases", []) if isinstance(payload, dict) else payload
    adapted: list[dict[str, Any]] = []
    for case in cases:
        paragraph = str(case.get("input_text") or "").strip()
        question = str(case.get("user_query") or "").strip()
        if not paragraph or not question:
            continue
        adapted.append(
            {
                "id": str(case.get("case_id") or case.get("name") or ""),
                "name": str(case.get("name") or ""),
                "category": str(case.get("category") or ""),
                "source_type": str(case.get("source_type") or ""),
                "paragraph": paragraph,
                "question": question,
                "expected_focus": case.get("expected_focus", {}),
            }
        )
    return adapted


def run_case(api_base: str, case: dict[str, Any]) -> dict[str, Any]:
    http_json("DELETE", f"{api_base}/reset", {"scope": "all"}, timeout=180)

    ingest_started = time.perf_counter()
    try:
        ingest = http_json(
            "POST",
            f"{api_base}/debug/ingest",
            {"texts": [case["paragraph"]]},
            timeout=900,
        )
        ingest_error = ""
    except Exception as exc:
        ingest, ingest_error = {}, f"{type(exc).__name__}: {exc}"
    ingest_seconds = time.perf_counter() - ingest_started

    atoms: list[str] = []
    quarantined = 0
    for item in ingest.get("results", []):
        for chunk in item.get("chunks", []):
            atoms.extend(chunk.get("atomspace_added", []))
            quarantined += sum(
                1
                for record in chunk.get("evidence_records", [])
                if record.get("validation_state") == "quarantined"
            )

    query_started = time.perf_counter()
    try:
        response = http_json(
            "POST",
            f"{api_base}/debug/query",
            {"question": case["question"]},
            timeout=420,
        )
        query_error = ""
    except Exception as exc:
        response, query_error = {}, f"{type(exc).__name__}: {exc}"
    query_seconds = time.perf_counter() - query_started

    langextract = response.get("langextract_postprocessed", {}) or {}
    return {
        "id": case["id"],
        "name": case["name"],
        "category": case["category"],
        "source_type": case["source_type"],
        "question": case["question"],
        "ingest_error": ingest_error,
        "ingest_seconds": round(ingest_seconds, 4),
        "atom_count": len(atoms),
        "quarantined_count": quarantined,
        "query_error": query_error,
        "translator_queries": langextract.get("queries", []),
        "provider_error": langextract.get("provider_error", ""),
        "pln_canonicalized_queries": response.get("pln_canonicalized_queries", []),
        "qdrant_aligned_queries": response.get("qdrant_aligned_queries", []),
        "execution_candidates": response.get("execution_candidates", []),
        "executed_query": response.get("executed_query", ""),
        "query_source": response.get("query_source", ""),
        "query_status": response.get("query_status", ""),
        "intent_mode": response.get("intent_mode", ""),
        "proof_status": response.get("proof_status", ""),
        "support_kind": response.get("support_kind", ""),
        "candidate_rejections": response.get("candidate_rejections", []),
        "rejection_reasons": response.get("rejection_reasons", []),
        "answer": response.get("answer", ""),
        "seconds": round(query_seconds, 4),
    }


def suite_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results) or 1
    executed = [r for r in results if r["executed_query"]]
    decisive = [r for r in results if r["proof_status"] in {"positive", "negative"}]
    no_query = [r for r in results if r["query_status"] == "no_query"]
    weak = [r for r in results if r["query_status"] == "weakly_aligned"]
    errors = [r for r in results if r["query_error"] or r["ingest_error"]]

    stages: dict[str, int] = {}
    for result in results:
        for rejection in result.get("candidate_rejections", []):
            stage = str(rejection.get("stage") or "unknown")
            stages[stage] = stages.get(stage, 0) + 1

    categories: dict[str, dict[str, int]] = {}
    for result in results:
        bucket = categories.setdefault(
            result["category"] or "uncategorized",
            {"total": 0, "executed": 0, "no_query": 0},
        )
        bucket["total"] += 1
        if result["executed_query"]:
            bucket["executed"] += 1
        if result["query_status"] == "no_query":
            bucket["no_query"] += 1

    # A translated query that never became an execution candidate is the exact
    # symptom reported in review: the parser produced something usable and a gate
    # discarded it.
    suppressed = [
        r
        for r in results
        if r["translator_queries"] and not r["execution_candidates"]
    ]

    return {
        "case_count": len(results),
        "executed_rate": round(len(executed) / total, 4),
        "no_query_rate": round(len(no_query) / total, 4),
        "weakly_aligned_rate": round(len(weak) / total, 4),
        "proof_found_rate": round(len(decisive) / total, 4),
        "suppressed_candidate_count": len(suppressed),
        "suppressed_case_ids": [r["id"] for r in suppressed],
        "request_error_count": len(errors),
        "request_error_case_ids": [r["id"] for r in errors],
        "rejection_stages": dict(sorted(stages.items(), key=lambda kv: -kv[1])),
        "by_category": categories,
        "intent_modes": _counter(r["intent_mode"] for r in results),
        "proof_status_counts": _counter(r["proof_status"] for r in results),
        "latency_seconds": {
            "ingest_p50": round(percentile([r["ingest_seconds"] for r in results], 0.50), 4),
            "ingest_p95": round(percentile([r["ingest_seconds"] for r in results], 0.95), 4),
            "query_p50": round(percentile([r["seconds"] for r in results], 0.50), 4),
            "query_p95": round(percentile([r["seconds"] for r in results], 0.95), 4),
        },
    }


def _counter(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "none")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default=os.getenv("PLN_RAG_API_BASE", "http://localhost:8000"))
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()

    print(BANNER)
    cases = load_cases(args.suite)
    if args.case_id:
        selected = set(args.case_id)
        cases = [case for case in cases if case["id"] in selected]
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()

    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']} {case['category']}")
        result = run_case(args.api_base.rstrip("/"), case)
        results.append(result)
        flag = "exec" if result["executed_query"] else "NO-QUERY"
        print(
            f"    {flag:9s} atoms={result['atom_count']:3d} "
            f"proof={result['proof_status'] or '-':9s} "
            f"mode={result['intent_mode'] or '-':11s} {result['seconds']:.1f}s"
        )
        if result["query_error"] or result["ingest_error"]:
            print(f"    ERROR {result['ingest_error']}{result['query_error']}")

    metrics = suite_metrics(results)
    report = {
        "created_at": started_at,
        "api_base": args.api_base,
        "suite_file": str(args.suite),
        "labels_available": False,
        "metrics": metrics,
        "cases": results,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = args.out_dir / f"stress_result_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    (args.out_dir / "latest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    print(f"\nsaved={out_path}")
    print(f"executed_rate      = {metrics['executed_rate']:.1%}")
    print(f"no_query_rate      = {metrics['no_query_rate']:.1%}")
    print(f"proof_found_rate   = {metrics['proof_found_rate']:.1%}")
    print(f"suppressed         = {metrics['suppressed_candidate_count']} "
          f"{metrics['suppressed_case_ids']}")
    print(f"request_errors     = {metrics['request_error_count']} "
          f"{metrics['request_error_case_ids']}")
    print(f"rejection_stages   = {metrics['rejection_stages']}")
    print("\nReminder: no labels in this suite. These are discovery metrics, not accuracy.")


if __name__ == "__main__":
    main()
