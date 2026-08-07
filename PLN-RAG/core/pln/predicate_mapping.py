from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Protocol

from core.pln.symbol_normalization import canonical_symbol


MAPPING_RELATIONS = {
    "exactMatch",
    "source_implies_target",
    "target_implies_source",
    "broader",
    "narrower",
    "related",
    "contradiction",
    "unrelated",
}
PROOF_SAFE_RELATIONS = {
    "exactMatch",
    "source_implies_target",
    "target_implies_source",
}
SOURCE_ROLES = {"fact", "conclusion"}
TARGET_ROLES = {"premise"}


@dataclass
class PredicateCard:
    predicate: str
    arity: int
    argument_types: list[str] = field(default_factory=list)
    definition: str = ""
    labels: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    source_atoms: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    origin: str = "current"

    @property
    def key(self) -> str:
        return f"{self.predicate}/{self.arity}"

    @property
    def search_text(self) -> str:
        parts = [
            f"predicate: {self.predicate}",
            f"label: {self.labels[0] if self.labels else humanize_predicate(self.predicate)}",
            f"arity: {self.arity}",
            f"argument types: {', '.join(self.argument_types)}",
            f"definition: {self.definition}",
        ]
        if self.examples:
            parts.append(f"evidence: {_truncate_text(self.examples[0], 180)}")
        if self.source_atoms:
            parts.append(f"atom: {self.source_atoms[0]}")
        return "; ".join(part for part in parts if part)

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "kind": "predicate_card",
                "key": self.key,
                "search_text": self.search_text,
            }
        )
        return payload

    def classifier_payload(self) -> dict[str, Any]:
        """Small evidence view used by the semantic classifier."""
        return {
            "predicate": self.predicate,
            "arity": self.arity,
            "argument_types": self.argument_types,
            "label": self.labels[0] if self.labels else humanize_predicate(
                self.predicate
            ),
            "definition": _truncate_text(self.definition, 180),
            "evidence": _truncate_text(self.examples[0], 240)
            if self.examples
            else "",
            "atom": self.source_atoms[0] if self.source_atoms else "",
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> PredicateCard | None:
        predicate = str(payload.get("predicate", "")).strip()
        try:
            arity = int(payload.get("arity", 0))
        except (TypeError, ValueError):
            return None
        if not predicate or arity <= 0:
            return None
        return cls(
            predicate=predicate,
            arity=arity,
            argument_types=_string_list(payload.get("argument_types")),
            definition=str(payload.get("definition", "")),
            labels=_string_list(payload.get("labels")),
            examples=_string_list(payload.get("examples")),
            source_atoms=_string_list(payload.get("source_atoms")),
            roles=_string_list(payload.get("roles")),
            origin=str(payload.get("origin", "registry")),
        )


@dataclass
class RelationProposal:
    relation: str
    confidence: float
    reason: str
    classifier: str = "unknown"
    argument_mapping: list[int] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidatedMapping:
    source: PredicateCard
    target: PredicateCard
    relation: str
    confidence: float
    reason: str
    classifier: str
    proof_safe: bool
    status: str
    evidence: dict[str, Any] = field(default_factory=dict)


class PredicateCardStore(Protocol):
    def upsert_predicate_cards(self, cards: list[dict[str, Any]]) -> None: ...

    def search_predicate_cards(
        self,
        search_text: str,
        *,
        arity: int,
        top_k: int,
        min_score: float,
    ) -> list[dict[str, Any]]: ...


class PredicateCardBuilder:
    def __init__(self, schema_aligner: Any):
        self.schema_aligner = schema_aligner

    def build(
        self,
        statements: list[str],
        *,
        source_text: str,
        origin: str,
    ) -> list[PredicateCard]:
        typed_arguments = self._collect_argument_types(statements)
        signatures: list[dict[str, Any]] = []
        for statement in statements:
            for signature in self.schema_aligner.extract_fact_signatures(statement):
                signatures.append({**signature, "role": "fact"})
            for signature in self.schema_aligner.extract_conclusion_signatures(statement):
                signatures.append({**signature, "role": "conclusion"})
            signatures.extend(
                self.schema_aligner.collect_premise_signatures(
                    [statement],
                    origin=origin,
                )
            )
            for signature in self.schema_aligner.extract_negated_fact_signatures(statement):
                signatures.append({**signature, "role": "negated_fact"})

        cards: dict[str, PredicateCard] = {}
        for signature in signatures:
            head = str(signature.get("head", ""))
            arity = int(signature.get("arity", 0) or 0)
            if not head or arity <= 0 or head in self.schema_aligner.skip_heads:
                continue
            key = f"{head}/{arity}"
            args = [str(arg) for arg in signature.get("args", [])]
            argument_types = [typed_arguments.get(arg, "entity") for arg in args]
            role = str(signature.get("role", "observed"))
            atom = self.schema_aligner.signature_atom(signature)
            card = cards.get(key)
            if card is None:
                label = humanize_predicate(head)
                card = PredicateCard(
                    predicate=head,
                    arity=arity,
                    argument_types=argument_types,
                    definition=self._definition(label, argument_types, source_text),
                    labels=[label],
                    examples=[source_text[:280]] if source_text else [],
                    source_atoms=[atom],
                    roles=[role],
                    origin=origin,
                )
                cards[key] = card
                continue
            card.argument_types = _merge_argument_types(
                card.argument_types,
                argument_types,
            )
            if atom not in card.source_atoms:
                card.source_atoms.append(atom)
            if role not in card.roles:
                card.roles.append(role)
            if source_text and source_text[:280] not in card.examples:
                card.examples.append(source_text[:280])
                card.examples = card.examples[-8:]
        return list(cards.values())

    def _collect_argument_types(self, statements: Iterable[str]) -> dict[str, str]:
        result: dict[str, str] = {}
        for statement in statements:
            for match in re.finditer(
                r"\(IsA\s+([^()\s]+)\s+([^()\s]+)\)",
                str(statement),
            ):
                result.setdefault(match.group(1), canonical_symbol(match.group(2)))
        return result

    def _definition(
        self,
        label: str,
        argument_types: list[str],
        source_text: str,
    ) -> str:
        signature = ", ".join(argument_types) if argument_types else "entity"
        return f"{label} is a predicate over {signature}."


class LLMPredicateRelationClassifier:
    """Classify predicate relationships; never decides proof safety itself."""

    GEMINI_MIN_TIMEOUT_SECONDS = 10

    def __init__(
        self,
        *,
        enabled: bool,
        gemini_api_key: str | None,
        gemini_model: str,
        openai_api_key: str | None,
        openai_model: str,
        timeout_seconds: int = 20,
    ):
        self.enabled = enabled
        self.gemini_api_key = gemini_api_key
        self.gemini_model = _strip_provider_prefix(gemini_model)
        self.openai_api_key = openai_api_key
        self.openai_model = _strip_provider_prefix(openai_model)
        self.timeout_seconds = max(self.GEMINI_MIN_TIMEOUT_SECONDS, int(timeout_seconds))

    @property
    def available(self) -> bool:
        return bool(
            self.enabled and (self.gemini_api_key or self.openai_api_key)
        )

    def classify(
        self,
        source: PredicateCard,
        target: PredicateCard,
    ) -> RelationProposal | None:
        return self.classify_batch([(source, target)])[0]

    def classify_batch(
        self,
        pairs: list[tuple[PredicateCard, PredicateCard]],
        *,
        timeout_seconds: int | None = None,
    ) -> list[RelationProposal | None]:
        """Classify a bounded candidate batch with at most one model call."""
        proposals: list[RelationProposal | None] = [None] * len(pairs)
        pending: list[tuple[int, PredicateCard, PredicateCard]] = []
        for index, (source, target) in enumerate(pairs):
            deterministic = self._deterministic_proposal(source, target)
            if deterministic is not None:
                proposals[index] = deterministic
            elif self.available:
                pending.append((index, source, target))

        if not pending:
            return proposals

        prompt = self._batch_prompt(pending)
        request_timeout = self.timeout_seconds
        if timeout_seconds is not None:
            request_timeout = max(
                1,
                min(self.timeout_seconds, int(timeout_seconds)),
            )
        if self.gemini_api_key and request_timeout < self.GEMINI_MIN_TIMEOUT_SECONDS:
            return proposals
        try:
            if self.gemini_api_key:
                raw_response = self._call_gemini(prompt, request_timeout)
                classifier = f"gemini:{self.gemini_model}"
            else:
                raw_response = self._call_openai(prompt, request_timeout)
                classifier = f"openai:{self.openai_model}"
            raw = _parse_json_object(raw_response)
            rows = raw.get("results", [])
            if not rows and len(pending) == 1 and "relation" in raw:
                rows = [{**raw, "id": pending[0][0]}]
            if not isinstance(rows, list):
                raise ValueError("classifier results must be an array")
            by_id = {
                int(row.get("id")): row
                for row in rows
                if isinstance(row, dict) and str(row.get("id", "")).isdigit()
            }
            for result_index, source, _target in pending:
                item = by_id.get(result_index)
                if not item:
                    continue
                relation = _normalize_relation(item.get("relation"))
                if relation not in MAPPING_RELATIONS:
                    continue
                argument_mapping = _integer_list(item.get("argument_mapping"))
                if not argument_mapping and source.arity == 1:
                    argument_mapping = [0]
                proposals[result_index] = RelationProposal(
                    relation=relation,
                    confidence=_clamp_confidence(item.get("confidence")),
                    reason=str(
                        item.get("reason", "model_classification")
                    ).strip()
                    or "model_classification",
                    classifier=classifier,
                    argument_mapping=argument_mapping,
                    raw=item,
                )
        except Exception as exc:
            names = ", ".join(
                f"{source.predicate}->{target.predicate}"
                for _, source, target in pending
            )
            print(
                "[PredicateRelationClassifier] Batch classification failed for "
                f"{names}: {exc}"
            )
        return proposals

    def _deterministic_proposal(
        self,
        source: PredicateCard,
        target: PredicateCard,
    ) -> RelationProposal | None:
        if source.predicate == target.predicate and source.arity == target.arity:
            return RelationProposal(
                relation="exactMatch",
                confidence=1.0,
                reason="identical_predicate_signature",
                classifier="deterministic",
                argument_mapping=list(range(source.arity)),
            )
        if set(_predicate_terms(source.predicate)) == set(
            _predicate_terms(target.predicate)
        ):
            return RelationProposal(
                relation="exactMatch",
                confidence=1.0,
                reason="identical_normalized_tokens_without_semantic_aliases",
                classifier="deterministic",
                argument_mapping=list(range(source.arity)),
            )

        return None

    def _batch_prompt(
        self,
        pending: list[tuple[int, PredicateCard, PredicateCard]],
    ) -> str:
        pairs = [
            {
                "id": index,
                "source": source.classifier_payload(),
                "target": target.classifier_payload(),
            }
            for index, source, target in pending
        ]
        return (
            "Classify each predicate pair for a formal proof system. "
            "Allowed relations: exactMatch, source_implies_target, "
            "target_implies_source, broader, narrower, related, "
            "contradiction, unrelated. Be conservative: topical similarity "
            "is not entailment. Preserve ordered arguments; argument_mapping "
            "uses zero-based positions and must be identity for a direct bridge.\n"
            "Examples of logical mappings:\n"
            "- Action to State/Attribute: If source is an action (e.g., 'EatsLargeMeals') and target is a state (e.g., 'HasHighCalorieIntake'), the relation is 'exactMatch' or 'source_implies_target'.\n"
            "- Semantic equivalence: 'UsesAutomobile' and 'DrivesCar' -> 'exactMatch'.\n"
            "- Specific to General: 'EatsApple' to 'ConsumesFruit' -> 'narrower'.\n"
            "Return exactly one result per id as JSON: "
            '{"results":[{"id":0,"relation":"...","confidence":0.0,'
            '"argument_mapping":[0],"reason":"..."}]}.\nPAIRS:\n'
            + json.dumps(pairs, separators=(",", ":"))
        )

    def _call_gemini(
        self,
        prompt: str,
        timeout_seconds: int | None = None,
    ) -> str:
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=self.gemini_api_key,
            http_options=types.HttpOptions(
                timeout=(timeout_seconds or self.timeout_seconds) * 1000
            ),
        )
        response = client.models.generate_content(
            model=self.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=1200,
                response_mime_type="application/json",
                response_json_schema=_batch_response_schema(),
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        parsed = getattr(response, "parsed", None)
        return parsed if isinstance(parsed, dict) else str(response.text or "")

    def _call_openai(
        self,
        prompt: str,
        timeout_seconds: int | None = None,
    ) -> str:
        from openai import OpenAI

        client = OpenAI(
            api_key=self.openai_api_key,
            timeout=float(timeout_seconds or self.timeout_seconds),
        )
        response = client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return strict JSON. Classify predicate semantics "
                        "conservatively for a formal proof system."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )
        return str(response.choices[0].message.content or "")


class PredicateMappingEngine:
    def __init__(
        self,
        *,
        classifier: LLMPredicateRelationClassifier,
        proof_threshold: float,
        retrieval_min_score: float,
        retrieval_top_k: int,
        max_candidates: int = 6,
        total_timeout_seconds: int = 25,
    ):
        self.classifier = classifier
        self.proof_threshold = proof_threshold
        self.retrieval_min_score = retrieval_min_score
        self.retrieval_top_k = retrieval_top_k
        self.max_candidates = max(1, int(max_candidates))
        self.total_timeout_seconds = max(1, int(total_timeout_seconds))
        self._failed_candidates: set[tuple[str, str, int]] = set()

    def discover(
        self,
        *,
        current_cards: list[PredicateCard],
        known_cards: list[PredicateCard],
        card_store: PredicateCardStore | None,
        known_mapping_keys: set[tuple[str, str, int]],
    ) -> tuple[list[ValidatedMapping], list[dict[str, Any]]]:
        decisions: list[dict[str, Any]] = []
        retrieved: dict[str, list[tuple[PredicateCard, float]]] = {}
        started_at = time.monotonic()
        self._failed_candidates.clear()

        def budget_exhausted() -> bool:
            return time.monotonic() - started_at >= self.total_timeout_seconds

        if card_store:
            try:
                card_store.upsert_predicate_cards(
                    [card.to_payload() for card in current_cards]
                )
                decisions.append(
                    {
                        "action": "predicate_cards_upserted",
                        "count": len(current_cards),
                    }
                )
                remote_search_enabled = bool(
                    getattr(self.classifier, "available", True)
                )
                for card in current_cards if remote_search_enabled else []:
                    if budget_exhausted():
                        decisions.append(
                            {
                                "action": "mapping_budget_exhausted",
                                "stage": "predicate_card_retrieval",
                                "budget_seconds": self.total_timeout_seconds,
                            }
                        )
                        break
                    matches = card_store.search_predicate_cards(
                        card.search_text,
                        arity=card.arity,
                        top_k=self.retrieval_top_k,
                        min_score=self.retrieval_min_score,
                    )
                    for match in matches:
                        candidate = PredicateCard.from_payload(
                            match.get("payload", {})
                        )
                        if not candidate or candidate.key == card.key:
                            continue
                        retrieved.setdefault(card.key, []).append(
                            (candidate, float(match.get("score", 0.0) or 0.0))
                        )
            except Exception as exc:
                decisions.append(
                    {
                        "action": "predicate_card_store_unavailable",
                        "reason": str(exc),
                    }
                )

        candidates: dict[
            tuple[str, str, int],
            tuple[PredicateCard, PredicateCard, float, str],
        ] = {}
        local_pool = _dedupe_cards(current_cards + known_cards)
        for current in current_cards:
            for other in local_pool:
                self._add_candidate(
                    candidates,
                    current,
                    other,
                    score=0.0,
                    source="registry_or_current_batch",
                )
            for other, score in retrieved.get(current.key, []):
                self._add_candidate(
                    candidates,
                    current,
                    other,
                    score=score,
                    source="qdrant_predicate_cards",
                )

        mappings: list[ValidatedMapping] = []
        ordered_candidates = [
            item
            for item in sorted(
                candidates.items(),
                key=lambda item: _candidate_priority(item[1]),
                reverse=True,
            )
            if item[0] not in known_mapping_keys
            and item[0] not in self._failed_candidates
        ][: self.max_candidates]
        if budget_exhausted():
            decisions.append(
                {
                    "action": "mapping_budget_exhausted",
                    "stage": "semantic_classification",
                    "budget_seconds": self.total_timeout_seconds,
                    "skipped_candidates": len(ordered_candidates),
                }
            )
            return mappings, decisions

        pairs = [(item[1][0], item[1][1]) for item in ordered_candidates]
        if hasattr(self.classifier, "classify_batch"):
            remaining_seconds = max(
                1,
                int(
                    self.total_timeout_seconds
                    - (time.monotonic() - started_at)
                ),
            )
            proposals = self.classifier.classify_batch(
                pairs,
                timeout_seconds=remaining_seconds,
            )
        else:
            proposals = [
                self.classifier.classify(source, target)
                for source, target in pairs
            ]

        for item, proposal in zip(ordered_candidates, proposals):
            key, (source, target, score, candidate_source) = item
            if proposal is None:
                self._failed_candidates.add(key)
                decisions.append(
                    {
                        "action": "mapping_candidate_unclassified",
                        "source": source.predicate,
                        "target": target.predicate,
                        "arity": source.arity,
                        "candidate_source": candidate_source,
                        "retrieval_score": score,
                        "reason": "no_classifier_result",
                    }
                )
                continue
            validated = self.validate(
                source,
                target,
                proposal,
                retrieval_score=score,
                candidate_source=candidate_source,
            )
            mappings.append(validated)
            decisions.append(
                {
                    "action": (
                        "mapping_approved"
                        if validated.proof_safe
                        else "mapping_recorded_not_proof_safe"
                    ),
                    "source": source.predicate,
                    "target": target.predicate,
                    "arity": source.arity,
                    "source_argument_types": source.argument_types,
                    "target_argument_types": target.argument_types,
                    "relation": validated.relation,
                    "confidence": validated.confidence,
                    "proof_safe": validated.proof_safe,
                    "status": validated.status,
                    "reason": validated.reason,
                    "classifier": validated.classifier,
                    "retrieval_score": score,
                    "candidate_source": candidate_source,
                    "argument_mapping": validated.evidence.get(
                        "argument_mapping",
                        [],
                    ),
                }
            )
        return mappings, decisions

    def validate(
        self,
        source: PredicateCard,
        target: PredicateCard,
        proposal: RelationProposal,
        *,
        retrieval_score: float,
        candidate_source: str,
    ) -> ValidatedMapping:
        reasons: list[str] = []
        relation = proposal.relation
        reason = proposal.reason
        if relation == "exactMatch" and proposal.classifier != "deterministic":
            # Candidate roles establish only the source -> target use needed by
            # the current proof. Do not grant a model bidirectional authority.
            relation = "source_implies_target"
            reason = f"{reason}; model_exact_match_narrowed_to_source_direction"
        if source.arity != target.arity or source.arity <= 0:
            reasons.append("arity_mismatch")
        if not argument_types_compatible(
            source.argument_types,
            target.argument_types,
        ):
            reasons.append("argument_type_mismatch")
        if relation not in PROOF_SAFE_RELATIONS:
            reasons.append("relation_not_proof_safe")
        if proposal.confidence < self.proof_threshold:
            reasons.append("confidence_below_threshold")
        expected_argument_mapping = list(range(source.arity))
        if proposal.argument_mapping != expected_argument_mapping:
            reasons.append("argument_order_not_identity")

        proof_safe = not reasons
        if reasons:
            reason = f"{reason}; gate_rejected: {', '.join(reasons)}"
        return ValidatedMapping(
            source=source,
            target=target,
            relation=relation,
            confidence=proposal.confidence,
            reason=reason,
            classifier=proposal.classifier,
            proof_safe=proof_safe,
            status="approved" if proof_safe else "candidate",
            evidence={
                "retrieval_score": retrieval_score,
                "candidate_source": candidate_source,
                "source_atom": source.source_atoms[:1],
                "target_atom": target.source_atoms[:1],
                "argument_mapping": proposal.argument_mapping,
            },
        )

    def _add_candidate(
        self,
        candidates: dict[
            tuple[str, str, int],
            tuple[PredicateCard, PredicateCard, float, str],
        ],
        left: PredicateCard,
        right: PredicateCard,
        *,
        score: float,
        source: str,
    ) -> None:
        if left.key == right.key or left.arity != right.arity:
            return
        oriented = _orient_cards(left, right)
        if not oriented:
            return
        source_card, target_card = oriented
        if not argument_types_compatible(
            source_card.argument_types,
            target_card.argument_types,
        ):
            return
        if score < self.retrieval_min_score and not _has_lexical_anchor(
            source_card,
            target_card,
        ):
            return
        key = (source_card.predicate, target_card.predicate, source_card.arity)
        previous = candidates.get(key)
        if previous and previous[2] >= score:
            return
        candidates[key] = (source_card, target_card, score, source)


def humanize_predicate(predicate: str) -> str:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", predicate)
    spaced = spaced.replace("_", " ")
    return " ".join(spaced.split()).lower()


def argument_types_compatible(left: list[str], right: list[str]) -> bool:
    if len(left) != len(right):
        return False
    for left_type, right_type in zip(left, right):
        if "entity" in {left_type, right_type}:
            continue
        if canonical_symbol(left_type) != canonical_symbol(right_type):
            return False
    return True


def mapping_allows_direction(
    *,
    mapping_source: str,
    mapping_target: str,
    relation: str,
    requested_source: str,
    requested_target: str,
) -> bool:
    if relation == "exactMatch":
        return {mapping_source, mapping_target} == {
            requested_source,
            requested_target,
        }
    if relation == "source_implies_target":
        return (
            mapping_source == requested_source
            and mapping_target == requested_target
        )
    if relation == "target_implies_source":
        return (
            mapping_target == requested_source
            and mapping_source == requested_target
        )
    return False


def _orient_cards(
    left: PredicateCard,
    right: PredicateCard,
) -> tuple[PredicateCard, PredicateCard] | None:
    left_roles = set(left.roles)
    right_roles = set(right.roles)
    if left_roles & SOURCE_ROLES and right_roles & TARGET_ROLES:
        return left, right
    if right_roles & SOURCE_ROLES and left_roles & TARGET_ROLES:
        return right, left
    return None


def _has_lexical_anchor(left: PredicateCard, right: PredicateCard) -> bool:
    left_terms = set(_predicate_terms(left.predicate))
    right_terms = set(_predicate_terms(right.predicate))
    return bool(left_terms & right_terms)


def _candidate_priority(
    candidate: tuple[PredicateCard, PredicateCard, float, str],
) -> tuple[float, float, int]:
    source, target, retrieval_score, candidate_source = candidate
    source_terms = set(_predicate_terms(source.predicate))
    target_terms = set(_predicate_terms(target.predicate))
    union = source_terms | target_terms
    lexical_score = len(source_terms & target_terms) / len(union) if union else 0.0
    qdrant_bonus = 1 if candidate_source == "qdrant_predicate_cards" else 0
    return retrieval_score, lexical_score, qdrant_bonus


def _predicate_terms(predicate: str) -> list[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", predicate)
    return [
        canonical_symbol(part)
        for part in re.split(r"[^A-Za-z0-9]+", spaced)
        if part and canonical_symbol(part) not in {"has", "have", "is", "of", "to"}
    ]


def _dedupe_cards(cards: Iterable[PredicateCard]) -> list[PredicateCard]:
    result: dict[str, PredicateCard] = {}
    for card in cards:
        result.setdefault(card.key, card)
    return list(result.values())


def _merge_argument_types(left: list[str], right: list[str]) -> list[str]:
    size = max(len(left), len(right))
    merged: list[str] = []
    for index in range(size):
        left_type = left[index] if index < len(left) else "entity"
        right_type = right[index] if index < len(right) else "entity"
        if left_type == "entity":
            merged.append(right_type)
        elif right_type == "entity" or left_type == right_type:
            merged.append(left_type)
        else:
            merged.append("entity")
    return merged


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _integer_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    result: list[int] = []
    for item in value:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            return []
    return result


def _normalize_relation(value: Any) -> str:
    normalized = str(value or "").strip()
    aliases = {
        "equivalent": "exactMatch",
        "exact_match": "exactMatch",
        "sourceImpliesTarget": "source_implies_target",
        "targetImpliesSource": "target_implies_source",
        "closeMatch": "related",
        "relatedMatch": "related",
    }
    return aliases.get(normalized, normalized)


def _clamp_confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def _parse_json_object(text: Any) -> dict[str, Any]:
    if isinstance(text, dict):
        return text
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("classifier did not return a JSON object")
    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("classifier response is not an object")
    return parsed


def _truncate_text(value: Any, limit: int) -> str:
    clean = " ".join(str(value or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + "..."


def _batch_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "relation": {
                            "type": "string",
                            "enum": sorted(MAPPING_RELATIONS),
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "argument_mapping": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "id",
                        "relation",
                        "confidence",
                        "argument_mapping",
                        "reason",
                    ],
                },
            }
        },
        "required": ["results"],
    }


def _strip_provider_prefix(model: str) -> str:
    """Kept as an alias; normalization lives in config so every consumer shares it."""
    from config import normalize_model_id

    return normalize_model_id(model) or ""
