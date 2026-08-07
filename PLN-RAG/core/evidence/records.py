from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from core.pln.symbol_normalization import canonical_symbol
from core.query.alignment import extract_query_targets
from core.query.intent import parse_query_signature


SCHEMA_VERSION = 2


def stable_id(prefix: str, *parts: object) -> str:
    material = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:32]}"


@dataclass(frozen=True)
class TargetDescriptor:
    target: str
    predicate: str
    arguments: tuple[str, ...]
    arity: int
    polarity: str
    has_variables: bool

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["arguments"] = list(self.arguments)
        return data


@dataclass(frozen=True)
class ClaimEvidenceRecord:
    document_id: str
    evidence_id: str
    claim_id: str
    claim_evidence_id: str
    atom: str
    original_text: str
    retrieval_text: str
    chunk_index: int
    chunk_start: int
    chunk_end: int
    source_start: int
    source_end: int
    global_source_start: int
    global_source_end: int
    validation_state: str
    validation_errors: tuple[str, ...] = ()
    lineage: dict[str, Any] = field(default_factory=dict)
    targets: tuple[TargetDescriptor, ...] = ()
    coreference: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    @property
    def is_reasoner_eligible(self) -> bool:
        return self.validation_state in {"accepted", "derived"}

    @property
    def is_retrieval_eligible(self) -> bool:
        return self.validation_state == "accepted" and bool(self.targets)

    @property
    def query_targets(self) -> list[str]:
        return [target.target for target in self.targets]

    def source_metadata(self) -> dict[str, Any]:
        return {
            "text": self.original_text,
            "char_interval": {
                "start_pos": self.source_start,
                "end_pos": self.source_end,
            },
            "global_char_interval": {
                "start_pos": self.global_source_start,
                "end_pos": self.global_source_end,
            },
            "evidence_id": self.evidence_id,
            "claim_id": self.claim_id,
            "claim_evidence_id": self.claim_evidence_id,
            "validation_state": self.validation_state,
            "lineage": self.lineage,
        }

    def ledger_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "evidence_id": self.evidence_id,
            "claim_id": self.claim_id,
            "claim_evidence_id": self.claim_evidence_id,
            "atom": self.atom,
            "original_text": self.original_text,
            "retrieval_text": self.retrieval_text,
            "chunk_index": self.chunk_index,
            "chunk_start": self.chunk_start,
            "chunk_end": self.chunk_end,
            "source_start": self.source_start,
            "source_end": self.source_end,
            "global_source_start": self.global_source_start,
            "global_source_end": self.global_source_end,
            "validation_state": self.validation_state,
            "validation_errors": list(self.validation_errors),
            "lineage": self.lineage,
            "targets": [target.to_dict() for target in self.targets],
            "coreference": self.coreference,
            "schema_version": self.schema_version,
        }

    def qdrant_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        if not self.is_retrieval_eligible:
            return records
        for target in self.targets:
            index_id = stable_id("index", self.claim_evidence_id, target.target)
            records.append(
                {
                    "index_id": index_id,
                    "claim_evidence_id": self.claim_evidence_id,
                    "document_id": self.document_id,
                    "evidence_id": self.evidence_id,
                    "claim_id": self.claim_id,
                    "nl": self.original_text,
                    "original_text": self.original_text,
                    "retrieval_text": self.retrieval_text,
                    "pln": [self.atom],
                    "query_targets": [target.target],
                    "predicate": target.predicate,
                    "arguments": list(target.arguments),
                    "entities": [
                        arg
                        for arg in target.arguments
                        if arg and not arg.startswith(("$", "?"))
                    ],
                    "arity": target.arity,
                    "polarity": target.polarity,
                    "has_variables": target.has_variables,
                    "validation_state": self.validation_state,
                    "source_start": self.global_source_start,
                    "source_end": self.global_source_end,
                    "lineage": self.lineage,
                    "schema_version": self.schema_version,
                }
            )
        return records


def build_claim_evidence_record(
    *,
    document_id: str,
    evidence_id: str,
    atom: str,
    chunk_text: str,
    chunk_index: int,
    chunk_start: int,
    source: dict[str, Any] | None,
    mention_prepass: dict[str, Any] | None = None,
) -> ClaimEvidenceRecord:
    clean_atom = " ".join(str(atom).split())
    source = dict(source or {})
    mention_prepass = dict(mention_prepass or {})
    source_text, local_start, local_end, errors = _validated_source_span(
        chunk_text,
        source,
    )

    if source:
        lineage = {
            "relation": source.get("lineage_relation", "extracted_from"),
            "score": source.get("lineage_score", 1.0),
            "extraction_class": source.get("class", ""),
        }
    else:
        source_text = chunk_text
        local_start = 0
        local_end = len(chunk_text)
        lineage = {
            "relation": "normalizer_generated",
            "score": 1.0,
            "extraction_class": "derived",
        }

    resolved = _resolved_mentions(mention_prepass, local_start, local_end)
    retrieval_text = _contextualized_text(source_text, resolved)
    targets = tuple(_target_descriptors(clean_atom))
    if source:
        errors.extend(
            _structural_validation_errors(
                targets=targets,
                retrieval_text=retrieval_text,
                extraction_class=str(source.get("class") or ""),
            )
        )
        validation_state = "accepted" if not errors else "quarantined"
    else:
        validation_state = "derived"
    if validation_state == "accepted" and not targets:
        errors.append("claim has no safe query target")
        validation_state = "quarantined"

    claim_id = stable_id("claim", clean_atom)
    claim_evidence_id = stable_id(
        "claim_evidence",
        claim_id,
        evidence_id,
        local_start,
        local_end,
    )
    return ClaimEvidenceRecord(
        document_id=document_id,
        evidence_id=evidence_id,
        claim_id=claim_id,
        claim_evidence_id=claim_evidence_id,
        atom=clean_atom,
        original_text=source_text,
        retrieval_text=retrieval_text,
        chunk_index=chunk_index,
        chunk_start=chunk_start,
        chunk_end=chunk_start + len(chunk_text),
        source_start=local_start,
        source_end=local_end,
        global_source_start=chunk_start + local_start,
        global_source_end=chunk_start + local_end,
        validation_state=validation_state,
        validation_errors=tuple(errors),
        lineage=lineage,
        targets=targets,
        coreference={"resolved_mentions": resolved},
    )


def normalize_query_target(target: str) -> tuple[str, str]:
    clean = " ".join(str(target).split())
    if clean.startswith("(Not ") and clean.endswith(")"):
        inner = clean[5:-1].strip()
        if inner.startswith("(") and inner.endswith(")") and _balanced(inner):
            return inner, "negative"
    return clean, "positive"


def _target_descriptors(atom: str) -> list[TargetDescriptor]:
    descriptors: list[TargetDescriptor] = []
    seen: set[tuple[str, str]] = set()
    for raw_target in extract_query_targets([atom]):
        target, polarity = normalize_query_target(raw_target)
        signature = parse_query_signature(f"(: $prf {target} $tv)")
        if not signature:
            continue
        key = (target, polarity)
        if key in seen:
            continue
        seen.add(key)
        arguments = tuple(str(arg) for arg in signature.get("args", []))
        descriptors.append(
            TargetDescriptor(
                target=target,
                predicate=str(signature["head"]),
                arguments=arguments,
                arity=int(signature["arity"]),
                polarity=polarity,
                has_variables="$" in target or "?" in target,
            )
        )
    return descriptors


def _validated_source_span(
    chunk_text: str,
    source: dict[str, Any],
) -> tuple[str, int, int, list[str]]:
    if not source:
        return chunk_text, 0, len(chunk_text), []

    errors: list[str] = []
    source_text = str(source.get("text", "")).strip()
    interval = source.get("char_interval") or {}
    start = interval.get("start_pos")
    end = interval.get("end_pos")

    if not source_text:
        return "", 0, 0, ["missing extraction source text"]

    if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(chunk_text):
        actual = chunk_text[start:end]
        if _normalized_text(actual) == _normalized_text(source_text):
            return actual.strip(), start, end, []

    occurrences = [match.start() for match in re.finditer(re.escape(source_text), chunk_text)]
    if len(occurrences) == 1:
        start = occurrences[0]
        end = start + len(source_text)
        return source_text, start, end, []

    if not occurrences:
        errors.append("extraction source is not an exact chunk substring")
    else:
        errors.append("extraction source span is ambiguous within the chunk")
    return source_text, 0, 0, errors


def _resolved_mentions(
    mention_prepass: dict[str, Any],
    start: int,
    end: int,
) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    mentions = mention_prepass.get("mentions", [])
    if not isinstance(mentions, list):
        return resolved
    for mention in mentions:
        if not isinstance(mention, dict) or mention.get("ambiguous"):
            continue
        resolved_to = str(mention.get("resolved_to") or "").strip()
        mention_start = mention.get("start")
        mention_end = mention.get("end")
        if (
            not resolved_to
            or not isinstance(mention_start, int)
            or not isinstance(mention_end, int)
            or mention_end <= start
            or mention_start >= end
        ):
            continue
        resolved.append(
            {
                "text": str(mention.get("text") or ""),
                "resolved_to": resolved_to,
                "source": str(mention.get("resolution_source") or ""),
                "cluster_id": mention.get("cluster_id"),
            }
        )
    return resolved


def _contextualized_text(text: str, resolved: list[dict[str, Any]]) -> str:
    if not resolved:
        return text
    annotations: list[str] = []
    seen: set[tuple[str, str]] = set()
    for mention in resolved:
        pair = (
            str(mention.get("text") or "").strip(),
            str(mention.get("resolved_to") or "").strip(),
        )
        if not all(pair) or pair in seen:
            continue
        seen.add(pair)
        annotations.append(f"{pair[0]} refers to {pair[1]}")
    if not annotations:
        return text
    return f"{text} Resolved context: {'; '.join(annotations)}."


def _structural_validation_errors(
    *,
    targets: tuple[TargetDescriptor, ...],
    retrieval_text: str,
    extraction_class: str,
) -> list[str]:
    errors: list[str] = []
    normalized_source = canonical_symbol(retrieval_text)
    source_tokens = {token for token in normalized_source.split("_") if token}
    source_compact = normalized_source.replace("_", "")

    for target in targets:
        if target.polarity == "negative" and not (
            extraction_class == "negative_fact"
            or _contains_explicit_negation(retrieval_text)
        ):
            errors.append("negative claim lacks explicit source negation")

        if target.has_variables:
            continue
        for argument in target.arguments:
            canonical_argument = canonical_symbol(argument)
            if not canonical_argument:
                continue
            compact_argument = canonical_argument.replace("_", "")
            argument_tokens = {
                token for token in canonical_argument.split("_") if token
            }
            if compact_argument in source_compact:
                continue
            if argument_tokens and argument_tokens.issubset(source_tokens):
                continue
            errors.append(f"ungrounded claim argument: {argument}")
    return list(dict.fromkeys(errors))


def _contains_explicit_negation(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:no|not|never|without|neither|nor|cannot|can't|"
            r"doesn't|didn't|isn't|wasn't|weren't|hasn't|haven't|lack|lacks|"
            r"lacking|absent)\b",
            text.lower(),
        )
    )


def _normalized_text(text: str) -> str:
    return " ".join(str(text).split())


def _balanced(text: str) -> bool:
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0
