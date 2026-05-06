import argparse
import asyncio
import json
import os
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

from config import get_settings
from core.service import PLNRAGService


SMOKE_CASES = [
    {
        "name": "dog-animal-yesno",
        "category": "isa",
        "texts": ["Dogs are animals.", "Fido is a dog."],
        "question": "Is Fido an animal?",
        "expected_proof": True,
    },
    {
        "name": "human-mortal-yesno",
        "category": "isa",
        "texts": ["Humans are mortal.", "Socrates is a human."],
        "question": "Is Socrates mortal?",
        "expected_proof": True,
    },
    {
        "name": "fish-smart-yesno",
        "category": "rule",
        "texts": ["People who eat fish are smart.", "Kebede eats fish."],
        "question": "Is Kebede smart?",
        "expected_proof": True,
    },
    {
        "name": "teacher-educates-yesno",
        "category": "rule",
        "texts": ["Teachers educate students.", "Marta is a teacher."],
        "question": "Does Marta educate students?",
        "expected_proof": True,
    },
    {
        "name": "programmer-solves-yesno",
        "category": "rule",
        "texts": ["Programmers solve problems.", "Bekele is a programmer."],
        "question": "Does Bekele solve problems?",
        "expected_proof": True,
    },
    {
        "name": "who-is-smart-open",
        "category": "open",
        "texts": ["People who eat fish are smart.", "Kebede eats fish."],
        "question": "Who is smart?",
        "expected_proof": True,
    },
    {
        "name": "what-does-kebede-eat-open",
        "category": "open",
        "texts": ["Kebede eats fish."],
        "question": "What does Kebede eat?",
        "expected_proof": True,
    },
    {
        "name": "hasa-dog-nose",
        "category": "relation",
        "texts": ["Dogs have noses.", "Fido is a dog."],
        "question": "Does Fido have a nose?",
        "expected_proof": True,
    },
    {
        "name": "usedfor-soap-cleaning",
        "category": "relation",
        "texts": ["Soap is used for cleaning."],
        "question": "Is soap used for cleaning?",
        "expected_proof": True,
    },
    {
        "name": "capableof-dog-drink-water",
        "category": "relation",
        "texts": ["Dogs are capable of drinking water."],
        "question": "Is dog capable of drinking water?",
        "expected_proof": True,
    },
    {
        "name": "partof-automobile-horn-car",
        "category": "relation",
        "texts": ["An automobile horn is part of a car."],
        "question": "Is automobile horn part of car?",
        "expected_proof": True,
    },
    {
        "name": "plural-dolphins-mammals",
        "category": "normalization",
        "texts": ["Dolphins are mammals."],
        "question": "Are dolphins mammals?",
        "expected_proof": True,
    },
    {
        "name": "negative-missing-fido",
        "category": "negative",
        "texts": ["Dogs are animals."],
        "question": "Is Fido an animal?",
        "expected_proof": False,
    },
    {
        "name": "negative-unrelated-query",
        "category": "negative",
        "texts": ["Cats are animals."],
        "question": "Is soap used for cleaning?",
        "expected_proof": False,
    },
]

SMOKE_QUICK_CASE_NAMES = {
    "dog-animal-yesno",
    "fish-smart-yesno",
    "what-does-kebede-eat-open",
    "usedfor-soap-cleaning",
    "negative-missing-fido",
}

CASE_FILES = {
    "entailmentbank": Path("data/benchmarks/cases/entailmentbank_actual_curated.json"),
    "abstracts": Path("data/benchmarks/cases/abstract_curated_candidates.json"),
}

ACTIVE_PARSERS = ("nl2pln", "canonical_pln")
AVAILABLE_PARSERS = (
    "nl2pln",
    "canonical_pln",
    "canonical_pln_1686527",
    "canonical_pln_d8d39afd",
)


def _slugify(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")


def _normalize_loaded_case(case: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(case)
    normalized.setdefault("name", normalized.get("id") or _slugify(normalized["question"]))
    if "texts" not in normalized:
        premises = normalized.get("premises", [])
        if not isinstance(premises, list):
            raise ValueError(f"Case {normalized['name']} has invalid premises field")
        normalized["texts"] = [str(item) for item in premises]
    normalized.setdefault("category", "external")
    if "expected_proof" not in normalized:
        raise ValueError(f"Case {normalized['name']} is missing expected_proof")
    return normalized


def _load_case_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cases"] = [_normalize_loaded_case(case) for case in payload.get("cases", [])]
    return payload


def _select_cases(suite: str, quick: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if suite == "smoke":
        cases = [
            case for case in SMOKE_CASES
            if not quick or case["name"] in SMOKE_QUICK_CASE_NAMES
        ]
        return (
            {
                "suite": "smoke",
                "status": "built_in",
                "source_type": "internal_smoke",
                "intended_use": "sanity_check",
            },
            cases,
        )

    if suite == "combined":
        entailmentbank = _load_case_file(CASE_FILES["entailmentbank"])
        abstracts = _load_case_file(CASE_FILES["abstracts"])
        cases = entailmentbank["cases"] + abstracts["cases"]
        if quick:
            cases = cases[:6]
        return (
            {
                "suite": "combined",
                "status": "composed",
                "source_type": "multi_suite",
                "members": [entailmentbank["suite"], abstracts["suite"]],
                "intended_use": "end_to_end_usefulness",
            },
            cases,
        )

    if suite not in CASE_FILES:
        raise ValueError(f"Unsupported suite '{suite}'")

    payload = _load_case_file(CASE_FILES[suite])
    cases = payload["cases"][:3] if quick else payload["cases"]
    metadata = {key: value for key, value in payload.items() if key != "cases"}
    return metadata, cases


def _get_parser_factory(name: str):
    if name == "nl2pln":
        from parsers.nl2pln_parser import NL2PLNParser

        return NL2PLNParser
    if name == "canonical_pln":
        from parsers.canonical_pln_parser import CanonicalPLNParser

        return CanonicalPLNParser
    if name == "canonical_pln_1686527":
        from parsers.canonical_pln_1686527_parser import CanonicalPLN1686527Parser

        return CanonicalPLN1686527Parser
    if name == "canonical_pln_d8d39afd":
        from parsers.canonical_pln_d8d39afd_parser import CanonicalPLND8D39AFDParser

        return CanonicalPLND8D39AFDParser
    if name == "manhin":
        from parsers.manhin_parser import ManhinParser

        return ManhinParser
    raise ValueError(f"Unsupported parser '{name}'")


def _run_parse(parser: object, text: str, context: list[str], is_query: bool):
    if is_query and hasattr(parser, "parse_query"):
        return parser.parse_query(text, context)
    return parser.parse(text, context)


def _proof_found(raw_proof: str) -> bool:
    return bool(raw_proof and raw_proof != "[]")


def _configure_case_environment(parser_name: str, case_name: str, run_id: str):
    slug = f"bench_{parser_name}_{case_name}_{run_id}".replace("-", "_")
    os.environ["QDRANT_COLLECTION"] = slug
    os.environ["ATOMSPACE_PATH"] = f"data/atomspace/{slug}.metta"
    os.environ["CONCEPTNET_ENABLED"] = "false"
    os.environ["CONCEPTNET_INDEX_ON_STARTUP"] = "false"
    os.environ["CONCEPTNET_AUTOLOAD"] = "false"
    os.environ["CONCEPTNET_AUTO_REBUILD_ON_CHANGE"] = "false"
    get_settings.cache_clear()
    return slug, Path(os.environ["ATOMSPACE_PATH"])


def _configure_suite_environment(parser_name: str, suite_name: str, run_id: str):
    slug = f"bench_{parser_name}_{suite_name}_{run_id}".replace("-", "_")
    os.environ["QDRANT_COLLECTION"] = slug
    os.environ["ATOMSPACE_PATH"] = f"data/atomspace/{slug}.metta"
    os.environ["CONCEPTNET_ENABLED"] = "false"
    os.environ["CONCEPTNET_INDEX_ON_STARTUP"] = "false"
    os.environ["CONCEPTNET_AUTOLOAD"] = "false"
    os.environ["CONCEPTNET_AUTO_REBUILD_ON_CHANGE"] = "false"
    get_settings.cache_clear()
    return slug, Path(os.environ["ATOMSPACE_PATH"])


def _compact_ingest_results(results) -> list[dict]:
    return [
        {
            "text": item.text,
            "status": item.status,
            "atoms": item.atoms,
            "error": item.error,
        }
        for item in results
    ]


async def _benchmark_case(parser_name: str, case: dict, run_id: str) -> dict:
    collection, atomspace_path = _configure_case_environment(parser_name, case["name"], run_id)
    parser_factory = _get_parser_factory(parser_name)

    init_started = time.perf_counter()
    parser = parser_factory()
    parser_init_seconds = time.perf_counter() - init_started

    parse_started = time.perf_counter()
    statement_parse = [_run_parse(parser, text, [], is_query=False) for text in case["texts"]]
    query_parse = _run_parse(parser, case["question"], [], is_query=True)
    parse_seconds = time.perf_counter() - parse_started

    service = PLNRAGService(parser)
    service.reset("all")

    try:
        ingest_started = time.perf_counter()
        ingest_results = await service.ingest_batch(case["texts"])
        ingest_seconds = time.perf_counter() - ingest_started

        query_started = time.perf_counter()
        query_response = await service.query(case["question"])
        query_seconds = time.perf_counter() - query_started

        total_seconds = parser_init_seconds + parse_seconds + ingest_seconds + query_seconds
        found = _proof_found(query_response.raw_proof)
        correct = found == case["expected_proof"]

        return {
            "case": case,
            "collection": collection,
            "atomspace_path": str(atomspace_path),
            "timing": {
                "parser_init_seconds": round(parser_init_seconds, 4),
                "parse_only_seconds": round(parse_seconds, 4),
                "ingest_seconds": round(ingest_seconds, 4),
                "query_seconds": round(query_seconds, 4),
                "total_seconds": round(total_seconds, 4),
            },
            "parse_only": {
                "statements": [result.statements for result in statement_parse],
                "statement_counts": [len(result.statements) for result in statement_parse],
                "query_statements": query_parse.statements,
                "query_statements_count": len(query_parse.statements),
                "queries": query_parse.queries,
                "query_count": len(query_parse.queries),
            },
            "end_to_end": {
                "ingest": _compact_ingest_results(ingest_results),
                "query": {
                    "question": query_response.question,
                    "pln_query": query_response.pln_query,
                    "original_query": query_response.original_query,
                    "executed_query": query_response.executed_query,
                    "fallback_used": query_response.fallback_used,
                    "query_status": query_response.query_status,
                    "raw_proof": query_response.raw_proof,
                    "sources": query_response.sources,
                    "answer": query_response.answer,
                },
            },
            "proof_found": found,
            "correct": correct,
        }
    finally:
        service.reset("all")
        if atomspace_path.exists():
            atomspace_path.unlink()


def _knowledge_state(service: PLNRAGService) -> dict[str, int]:
    info = service.health()
    return {
        "atomspace_size": int(info.get("atomspace_size", 0)),
        "background_atomspace_size": int(info.get("background_atomspace_size", 0)),
        "vectordb_count": int(info.get("vectordb_count", 0)),
    }


async def _benchmark_case_with_service(
    parser_name: str,
    case: dict,
    run_id: str,
    collection: str,
    atomspace_path: Path,
    parser: object,
    service: PLNRAGService,
) -> dict:
    parse_started = time.perf_counter()
    statement_parse = [_run_parse(parser, text, [], is_query=False) for text in case["texts"]]
    query_parse = _run_parse(parser, case["question"], [], is_query=True)
    parse_seconds = time.perf_counter() - parse_started

    state_before = _knowledge_state(service)

    ingest_started = time.perf_counter()
    ingest_results = await service.ingest_batch(case["texts"])
    ingest_seconds = time.perf_counter() - ingest_started

    query_started = time.perf_counter()
    query_response = await service.query(case["question"])
    query_seconds = time.perf_counter() - query_started

    total_seconds = parse_seconds + ingest_seconds + query_seconds
    found = _proof_found(query_response.raw_proof)
    correct = found == case["expected_proof"]
    state_after = _knowledge_state(service)

    return {
        "case": case,
        "collection": collection,
        "atomspace_path": str(atomspace_path),
        "timing": {
            "parser_init_seconds": 0.0,
            "parse_only_seconds": round(parse_seconds, 4),
            "ingest_seconds": round(ingest_seconds, 4),
            "query_seconds": round(query_seconds, 4),
            "total_seconds": round(total_seconds, 4),
        },
        "knowledge_state_before_case": state_before,
        "knowledge_state_after_case": state_after,
        "parse_only": {
            "statements": [result.statements for result in statement_parse],
            "statement_counts": [len(result.statements) for result in statement_parse],
            "query_statements": query_parse.statements,
            "query_statements_count": len(query_parse.statements),
            "queries": query_parse.queries,
            "query_count": len(query_parse.queries),
        },
        "end_to_end": {
            "ingest": _compact_ingest_results(ingest_results),
            "query": {
                "question": query_response.question,
                "pln_query": query_response.pln_query,
                "original_query": query_response.original_query,
                "executed_query": query_response.executed_query,
                "fallback_used": query_response.fallback_used,
                "query_status": query_response.query_status,
                "raw_proof": query_response.raw_proof,
                "sources": query_response.sources,
                "answer": query_response.answer,
            },
        },
        "proof_found": found,
        "correct": correct,
    }


async def _benchmark_parser_cumulative(
    parser_name: str,
    cases: list[dict[str, Any]],
    run_id: str,
    suite_name: str,
) -> list[dict[str, Any]]:
    collection, atomspace_path = _configure_suite_environment(parser_name, suite_name, run_id)
    parser_factory = _get_parser_factory(parser_name)

    init_started = time.perf_counter()
    parser = parser_factory()
    parser_init_seconds = time.perf_counter() - init_started
    service = PLNRAGService(parser)
    service.reset("all")

    results: list[dict[str, Any]] = []
    try:
        for index, case in enumerate(cases):
            result = await _benchmark_case_with_service(
                parser_name,
                case,
                run_id,
                collection,
                atomspace_path,
                parser,
                service,
            )
            if index == 0:
                result["timing"]["parser_init_seconds"] = round(parser_init_seconds, 4)
                result["timing"]["total_seconds"] = round(
                    result["timing"]["total_seconds"] + parser_init_seconds, 4
                )
            results.append(result)
    finally:
        service.reset("all")
        if atomspace_path.exists():
            atomspace_path.unlink()

    return results


def _summarize_parser(results: list[dict]) -> dict:
    total_cases = len(results)
    correct = sum(1 for result in results if result.get("correct"))
    proof_found = sum(1 for result in results if result.get("proof_found"))
    no_query = sum(
        1
        for result in results
        if result.get("end_to_end", {}).get("query", {}).get("query_status") == "no_query"
    )
    weakly_aligned = sum(
        1
        for result in results
        if result.get("end_to_end", {}).get("query", {}).get("query_status") == "weakly_aligned"
    )
    fallback_used = sum(
        1
        for result in results
        if result.get("end_to_end", {}).get("query", {}).get("fallback_used")
    )
    latencies = [result.get("timing", {}).get("total_seconds", 0.0) for result in results]
    return {
        "cases": total_cases,
        "correct": correct,
        "proof_found": proof_found,
        "no_query": no_query,
        "weakly_aligned": weakly_aligned,
        "fallback_used": fallback_used,
        "avg_latency_seconds": round(sum(latencies) / total_cases, 4) if total_cases else 0.0,
        "median_latency_seconds": round(statistics.median(latencies), 4) if latencies else 0.0,
    }


def _markdown_summary(summary: dict[str, dict]) -> str:
    lines = [
        "| Parser | Cases | Correct | Proof Found | No Query | Weak Align | Fallback | Avg Latency | Median Latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for parser_name, stats in summary.items():
        lines.append(
            f"| {parser_name} | {stats['cases']} | {stats['correct']} | {stats['proof_found']} | "
            f"{stats['no_query']} | {stats['weakly_aligned']} | {stats['fallback_used']} | "
            f"{stats['avg_latency_seconds']:.4f}s | {stats['median_latency_seconds']:.4f}s |"
        )
    return "\n".join(lines)


async def main() -> int:
    cli = argparse.ArgumentParser(description="Benchmark all PLN-RAG parsers without ConceptNet.")
    cli.add_argument(
        "--mode",
        choices=("isolated", "cumulative"),
        default="isolated",
        help="Whether to reset per case or accumulate knowledge per parser across a suite",
    )
    cli.add_argument(
        "--suite",
        choices=("smoke", "entailmentbank", "abstracts", "combined"),
        default="combined",
        help="Benchmark suite to run",
    )
    cli.add_argument(
        "--parsers",
        nargs="+",
        choices=AVAILABLE_PARSERS,
        default=list(ACTIVE_PARSERS),
        help="Parsers to include in this benchmark run",
    )
    cli.add_argument("--quick", action="store_true", help="Run a reduced representative case set")
    cli.add_argument(
        "--output-dir",
        default="data/benchmarks",
        help="Directory where benchmark JSON reports are written",
    )
    args = cli.parse_args()

    suite_metadata, cases = _select_cases(args.suite, args.quick)
    run_id = uuid.uuid4().hex[:8]
    payload: dict[str, object] = {
        "run_id": run_id,
        "conceptnet_enabled": False,
        "mode": args.mode,
        "suite": args.suite,
        "suite_metadata": suite_metadata,
        "case_count": len(cases),
        "parsers": {},
        "summary": {},
    }

    payload["active_parsers"] = list(args.parsers)

    for parser_name in args.parsers:
        results = []
        if args.mode == "cumulative":
            try:
                results = await _benchmark_parser_cumulative(
                    parser_name,
                    cases,
                    run_id,
                    args.suite,
                )
            except Exception as exc:
                for case in cases:
                    results.append(
                        {
                            "case": case,
                            "error": str(exc),
                            "proof_found": False,
                            "correct": False,
                            "timing": {
                                "parser_init_seconds": 0.0,
                                "parse_only_seconds": 0.0,
                                "ingest_seconds": 0.0,
                                "query_seconds": 0.0,
                                "total_seconds": 0.0,
                            },
                            "end_to_end": {
                                "query": {"query_status": "error", "fallback_used": False}
                            },
                        }
                    )
        else:
            for case in cases:
                try:
                    result = await _benchmark_case(parser_name, case, run_id)
                except Exception as exc:
                    result = {
                        "case": case,
                        "error": str(exc),
                        "proof_found": False,
                        "correct": False,
                        "timing": {
                            "parser_init_seconds": 0.0,
                            "parse_only_seconds": 0.0,
                            "ingest_seconds": 0.0,
                            "query_seconds": 0.0,
                            "total_seconds": 0.0,
                        },
                        "end_to_end": {"query": {"query_status": "error", "fallback_used": False}},
                    }
                results.append(result)
        payload["parsers"][parser_name] = results
        payload["summary"][parser_name] = _summarize_parser(results)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"parser_benchmark_{_slugify(args.suite)}_{_slugify(args.mode)}_{run_id}.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps({"output": str(output_path), "summary": payload["summary"]}, indent=2))
    print()
    print(_markdown_summary(payload["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
