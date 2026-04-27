import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sys
from collections import Counter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.symbol_normalization import NORMALIZATION_VERSION, canonical_symbol


DEFAULT_INPUT_FILE = "data/conceptnet/conceptnet-assertions-5.7.0.csv.gz"
ATOMSPACE_OUTPUT = "data/conceptnet/conceptnet_background.metta"
VECTOR_OUTPUT = "data/conceptnet/conceptnet_background.jsonl"
MANIFEST_OUTPUT = "data/conceptnet/conceptnet_manifest.json"
LANGUAGE_FILTER = "/c/en/"
MIN_WEIGHT = 2.0
DEFAULT_COVERAGE_PERCENT = 100.0
DEFAULT_SAMPLE_SEED = 42

RELATION_MAP = {
    "/r/IsA": "IsA",
    "/r/HasProperty": "IsA",
    "/r/PartOf": "PartOf",
    "/r/UsedFor": "UsedFor",
    "/r/CapableOf": "CapableOf",
    "/r/AtLocation": "AtLocation",
    "/r/HasA": "HasA",
    "/r/MadeOf": "MadeOf",
}

NL_TEMPLATES = {
    "IsA": "{s} is a kind of {o}",
    "PartOf": "{s} is part of {o}",
    "UsedFor": "{s} is used for {o}",
    "CapableOf": "{s} is capable of {o}",
    "AtLocation": "{s} is located at {o}",
    "HasA": "{s} has a {o}",
    "MadeOf": "{s} is made of {o}",
}


def clean_concept(uri: str) -> str | None:
    if not uri.startswith(LANGUAGE_FILTER):
        return None
    text = uri[len(LANGUAGE_FILTER) :].split("/")[0]
    return canonical_symbol(text, lemmatize=True)


def calculate_stv(weight: float) -> str:
    strength = 1.0
    confidence = weight / (weight + 0.2)
    confidence = min(confidence, 0.9999)
    return f"(STV {strength} {confidence:.4f})"


def generate_id(predicate: str, source: str, target: str) -> str:
    digest = hashlib.md5(f"{predicate}|{source}|{target}".encode("utf-8")).hexdigest()
    return f"cnet_{predicate.lower()}_{digest[:10]}"


def make_nl(predicate: str, source: str, target: str, original_relation: str) -> str:
    if original_relation == "/r/HasProperty":
        return f"{source} is {target}"
    return NL_TEMPLATES[predicate].format(s=source.replace("_", " "), o=target.replace("_", " "))


def sha256_of_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=os.getenv("CONCEPTNET_INPUT_FILE", DEFAULT_INPUT_FILE),
    )
    parser.add_argument(
        "--atomspace-output",
        default=os.getenv("CONCEPTNET_ATOMSPACE_PATH", ATOMSPACE_OUTPUT),
    )
    parser.add_argument(
        "--vector-output",
        default=os.getenv("CONCEPTNET_VECTOR_PAYLOAD_PATH", VECTOR_OUTPUT),
    )
    parser.add_argument(
        "--manifest-output",
        default=os.getenv("CONCEPTNET_MANIFEST_PATH", MANIFEST_OUTPUT),
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=float(os.getenv("CONCEPTNET_MIN_WEIGHT", str(MIN_WEIGHT))),
    )
    parser.add_argument(
        "--coverage-percent",
        type=float,
        default=float(
            os.getenv(
                "CONCEPTNET_COVERAGE_PERCENT",
                str(DEFAULT_COVERAGE_PERCENT),
            )
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=int(os.getenv("CONCEPTNET_SAMPLE_SEED", str(DEFAULT_SAMPLE_SEED))),
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Maximum number of source rows to scan before stopping (0 = no limit)",
    )
    return parser.parse_args()


def sampling_score(predicate: str, source: str, target: str, seed: int) -> float:
    key = f"{seed}|{predicate}|{source}|{target}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def should_keep(predicate: str, source: str, target: str, coverage_percent: float, seed: int) -> bool:
    if coverage_percent >= 100.0:
        return True
    if coverage_percent <= 0.0:
        return False
    threshold = coverage_percent / 100.0
    return sampling_score(predicate, source, target, seed) < threshold


def main():
    args = parse_args()
    if args.coverage_percent <= 0 or args.coverage_percent > 100:
        raise ValueError("coverage percent must be in the range (0, 100]")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    os.makedirs(os.path.dirname(args.atomspace_output), exist_ok=True)

    seen: set[tuple[str, str, str]] = set()
    relation_counts: Counter[str] = Counter()
    atom_count = 0
    scanned_rows = 0
    eligible_row_count = 0
    sampled_row_count = 0

    with gzip.open(args.input, "rt", encoding="utf-8") as src, open(
        args.atomspace_output, "w", encoding="utf-8"
    ) as atom_out, open(args.vector_output, "w", encoding="utf-8") as vector_out:
        reader = csv.reader(src, delimiter="\t")
        for row in reader:
            scanned_rows += 1
            if args.max_rows and scanned_rows > args.max_rows:
                break
            if len(row) < 5:
                continue
            _, relation, start, end, metadata_json = row
            predicate = RELATION_MAP.get(relation)
            if not predicate:
                continue

            source = clean_concept(start)
            target = clean_concept(end)
            if not source or not target:
                continue

            try:
                metadata = json.loads(metadata_json)
                weight = float(metadata.get("weight", 1.0))
            except Exception:
                continue
            if weight < args.min_weight:
                continue

            eligible_row_count += 1
            if not should_keep(
                predicate, source, target, args.coverage_percent, args.sample_seed
            ):
                continue
            sampled_row_count += 1

            triple = (predicate, source, target)
            if triple in seen:
                continue
            seen.add(triple)

            stmt_id = generate_id(predicate, source, target)
            atom = f"(: {stmt_id} ({predicate} {source} {target}) {calculate_stv(weight)})"
            atom_out.write(atom + "\n")

            payload = {
                "nl": make_nl(predicate, source, target, relation),
                "pln": [atom],
                "source": "conceptnet",
                "background": True,
                "relation": predicate,
            }
            vector_out.write(json.dumps(payload, ensure_ascii=True) + "\n")

            atom_count += 1
            relation_counts[predicate] += 1

    manifest = {
        "source_file": args.input,
        "normalization_version": NORMALIZATION_VERSION,
        "min_weight": args.min_weight,
        "coverage_percent": args.coverage_percent,
        "sample_seed": args.sample_seed,
        "scanned_rows": scanned_rows,
        "max_rows": args.max_rows,
        "eligible_row_count": eligible_row_count,
        "sampled_row_count": sampled_row_count,
        "deduped_atom_count": atom_count,
        "atom_count": atom_count,
        "vector_record_count": atom_count,
        "relation_counts": dict(sorted(relation_counts.items())),
        "atomspace_sha256": sha256_of_file(args.atomspace_output),
        "vector_payload_sha256": sha256_of_file(args.vector_output),
    }
    with open(args.manifest_output, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
