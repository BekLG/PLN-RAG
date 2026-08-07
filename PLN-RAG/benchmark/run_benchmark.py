from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmark"
DEFAULT_CASES = BENCHMARK_DIR / "cases.json"
DEFAULT_RESULTS_DIR = BENCHMARK_DIR / "results"


def http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 600,
) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed: {exc.code} {body}") from exc


def status_matches(expected: str, observed: str) -> bool:
    return expected == observed


def failure_category(expected: str, observed: str, response: dict[str, Any]) -> str:
    if expected == observed:
        return "correct"
    if not response.get("executed_query"):
        return "no_query"
    if response.get("target_alignment") == "rejected" or response.get("query_status") == "weakly_aligned":
        return "invalid_target"
    if expected in {"unknown", "unanswered"} and observed in {"positive", "negative", "both"}:
        return "overclaim"
    if expected in {"positive", "negative"} and observed in {"unknown", "unanswered"}:
        return f"missed_{expected}"
    return "wrong_polarity"


def run_case(api_base: str, case: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    http_json("DELETE", f"{api_base}/reset", {"scope": "all"}, timeout=120)

    ingest_started = time.perf_counter()
    ingest = http_json(
        "POST",
        f"{api_base}/debug/ingest",
        {"texts": [case["paragraph"]]},
        timeout=900,
    )
    ingest_seconds = time.perf_counter() - ingest_started

    atoms: list[str] = []
    chunks: list[dict[str, Any]] = []
    for item in ingest.get("results", []):
        for chunk in item.get("chunks", []):
            atoms.extend(chunk.get("atomspace_added", []))
            chunks.append(
                {
                    "chunk": chunk.get("chunk", ""),
                    "atom_count": len(chunk.get("atomspace_added", [])),
                    "atoms": chunk.get("atomspace_added", []),
                    "schema_alignment": chunk.get("schema_alignment", []),
                    "predicate_registry": chunk.get("predicate_registry", []),
                }
            )

    query_results: list[dict[str, Any]] = []
    for query_case in case.get("queries", []):
        query_started = time.perf_counter()
        response = http_json(
            "POST",
            f"{api_base}/debug/query",
            {"question": query_case["question"]},
            timeout=420,
        )
        query_seconds = time.perf_counter() - query_started
        observed = response.get("proof_status", "unknown")
        expected = query_case.get("expected_status", "")
        query_results.append(
            {
                "question": query_case["question"],
                "expected_status": expected,
                "observed_status": observed,
                "matched": status_matches(expected, observed),
                "expected_answer": query_case.get("expected_answer", ""),
                "rationale": query_case.get("rationale", ""),
                "answer": response.get("answer", ""),
                "executed_query": response.get("executed_query", ""),
                "query_source": response.get("query_source", ""),
                "query_status": response.get("query_status", ""),
                "intent_mode": response.get("intent_mode", ""),
                "negative_query": response.get("negative_query", ""),
                "execution_candidates": response.get("execution_candidates", []),
                "positive_proof": response.get("positive_proof", []),
                "negative_proof": response.get("negative_proof", []),
                "requirements": response.get("requirements", []),
                "canonical_proposition": response.get("canonical_proposition", ""),
                "target_alignment": response.get("target_alignment", "none"),
                "proof_validated": response.get("proof_validated", False),
                "support_kind": response.get("support_kind", "unknown"),
                "unresolved_mentions": response.get("unresolved_mentions", []),
                "normalization_evidence": response.get("normalization_evidence", []),
                "rejection_reasons": response.get("rejection_reasons", []),
                # Per-candidate gate audit. Without this a no_query result cannot
                # be attributed to the gate that caused it.
                "candidate_rejections": response.get("candidate_rejections", []),
                "translator_queries": (
                    response.get("langextract_postprocessed", {}) or {}
                ).get("queries", []),
                "pln_canonicalized_queries": response.get("pln_canonicalized_queries", []),
                "failure_category": failure_category(expected, observed, response),
                "seconds": round(query_seconds, 4),
            }
        )

    matched = sum(1 for item in query_results if item["matched"])
    total = len(query_results)
    return {
        "id": case["id"],
        "topic": case.get("topic", ""),
        "paragraph": case.get("paragraph", ""),
        "ingest_status": ingest.get("results", [{}])[0].get("status", "unknown"),
        "ingest_seconds": round(ingest_seconds, 4),
        "atom_count": len(atoms),
        "chunks": chunks,
        "queries": query_results,
        "matched": matched,
        "total": total,
        "accuracy": round(matched / total, 4) if total else 0.0,
        "seconds": round(time.perf_counter() - started, 4),
    }


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile_value
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def benchmark_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    queries = [query for case in results for query in case.get("queries", [])]
    direct = [query for query in queries if query["expected_status"] != "unanswered"]
    decisive = [
        query
        for query in queries
        if query["observed_status"] in {"positive", "negative"}
    ]
    expected_groups: dict[str, dict[str, Any]] = {}
    for status in ("positive", "negative", "unknown", "unanswered"):
        group = [query for query in queries if query["expected_status"] == status]
        matched = sum(1 for query in group if query["matched"])
        expected_groups[status] = {
            "total": len(group),
            "matched": matched,
            "accuracy": round(matched / len(group), 4) if group else 0.0,
        }
    populated = [group["accuracy"] for group in expected_groups.values() if group["total"]]
    ingest_times = [float(case.get("ingest_seconds", 0.0)) for case in results]
    query_times = [float(query.get("seconds", 0.0)) for query in queries]
    categories: dict[str, int] = {}
    for query in queries:
        category = query.get("failure_category", "unknown")
        categories[category] = categories.get(category, 0) + 1
    decisive_correct = sum(1 for query in decisive if query["matched"])
    validated_proofs = [
        query for query in decisive if query.get("proof_validated") is True
    ]
    # Which gate discarded candidates, and how often a usable parser query was
    # produced but no candidate survived to execution.
    stages: dict[str, int] = {}
    for query in queries:
        for rejection in query.get("candidate_rejections", []) or []:
            stage = str(rejection.get("stage") or "unknown")
            stages[stage] = stages.get(stage, 0) + 1
    suppressed = [
        query
        for query in queries
        if query.get("translator_queries")
        and not query.get("execution_candidates")
    ]
    return {
        "rejection_stages": dict(sorted(stages.items(), key=lambda kv: -kv[1])),
        "suppressed_candidate_count": len(suppressed),
        "by_expected_status": expected_groups,
        "macro_status_accuracy": round(statistics.mean(populated), 4) if populated else 0.0,
        "direct_query_count": len(direct),
        "direct_query_accuracy": round(
            sum(1 for query in direct if query["matched"]) / len(direct),
            4,
        ) if direct else 0.0,
        "decisive_answer_count": len(decisive),
        "decisive_answer_precision": round(decisive_correct / len(decisive), 4) if decisive else 0.0,
        "validated_decisive_proof_rate": round(len(validated_proofs) / len(decisive), 4) if decisive else 1.0,
        "overclaim_count": categories.get("overclaim", 0),
        "no_query_count": categories.get("no_query", 0),
        "invalid_target_count": categories.get("invalid_target", 0),
        "failure_categories": categories,
        "latency_seconds": {
            "ingest_p50": round(percentile(ingest_times, 0.50), 4),
            "ingest_p95": round(percentile(ingest_times, 0.95), 4),
            "query_p50": round(percentile(query_times, 0.50), 4),
            "query_p95": round(percentile(query_times, 0.95), 4),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default=os.getenv("PLN_RAG_API_BASE", "http://localhost:8000"))
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if args.case_id:
        selected = set(args.case_id)
        cases = [case for case in cases if case.get("id") in selected]
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()

    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']} - ingest/query")
        results.append(run_case(args.api_base.rstrip("/"), case))

    total_queries = sum(item["total"] for item in results)
    matched_queries = sum(item["matched"] for item in results)
    report = {
        "created_at": started_at,
        "api_base": args.api_base,
        "cases_file": str(args.cases),
        "case_count": len(results),
        "query_count": total_queries,
        "matched_count": matched_queries,
        "accuracy": round(matched_queries / total_queries, 4) if total_queries else 0.0,
        "metrics": benchmark_metrics(results),
        "cases": results,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = args.out_dir / f"benchmark_result_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    latest_path = args.out_dir / "latest.json"
    latest_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"saved={out_path}")
    print(f"accuracy={matched_queries}/{total_queries} ({report['accuracy']:.1%})")
    print(
        "direct_accuracy="
        f"{report['metrics']['direct_query_accuracy']:.1%} "
        f"overclaims={report['metrics']['overclaim_count']} "
        f"no_query={report['metrics']['no_query_count']}"
    )


if __name__ == "__main__":
    main()
