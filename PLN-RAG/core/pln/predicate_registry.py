from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, List

from core.pln.predicate_mapping import (
    PROOF_SAFE_RELATIONS,
    PredicateCard,
    PredicateCardBuilder,
    PredicateCardStore,
    PredicateMappingEngine,
    ValidatedMapping,
    argument_types_compatible,
    humanize_predicate,
    mapping_allows_direction,
)
from core.pln.schema_alignment import PLNSchemaAligner


@dataclass
class PredicateEntry:
    predicate: str
    arity: int
    terms: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    argument_types: List[str] = field(default_factory=list)
    definitions: List[str] = field(default_factory=list)
    source_atoms: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)
    status: str = "observed"
    source_count: int = 0


@dataclass
class PredicateMapping:
    source: str
    target: str
    arity: int
    relation: str
    confidence: float
    reason: str
    proof_safe: bool = False
    status: str = "candidate"
    classifier: str = "unknown"
    source_argument_types: List[str] = field(default_factory=list)
    target_argument_types: List[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


class PredicateRegistry:
    """Persistent predicate cards and validated, directional mappings."""

    def __init__(
        self,
        *,
        path: str | Path | None = None,
        schema_aligner: PLNSchemaAligner,
        mapping_engine: PredicateMappingEngine | None = None,
        autosave: bool = True,
    ):
        self.path = Path(path) if path else None
        self.schema_aligner = schema_aligner
        self.mapping_engine = mapping_engine
        self.card_builder = PredicateCardBuilder(schema_aligner)
        self.card_store: PredicateCardStore | None = None
        self.autosave = bool(autosave and self.path)
        self.entries: dict[str, PredicateEntry] = {}
        self.mappings: dict[tuple[str, str, int], PredicateMapping] = {}
        self.load()

    def set_card_store(self, card_store: PredicateCardStore | None) -> None:
        self.card_store = card_store

    def load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return

        for item in data.get("predicates", []):
            try:
                entry = PredicateEntry(**item)
            except TypeError:
                continue
            self.entries[self.entry_key(entry.predicate, entry.arity)] = entry

        for item in data.get("mappings", []):
            if item.get("relation") == "implies":
                item = {**item, "relation": "source_implies_target"}
            try:
                mapping = PredicateMapping(**item)
            except TypeError:
                continue
            self.mappings[self.mapping_key(
                mapping.source,
                mapping.target,
                mapping.arity,
            )] = mapping

    def save(self) -> None:
        if not self.autosave or not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 2,
            "predicates": [asdict(entry) for entry in self.entries.values()],
            "mappings": [asdict(mapping) for mapping in self.mappings.values()],
        }
        self.path.write_text(
            json.dumps(data, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def clear(self) -> None:
        self.entries.clear()
        self.mappings.clear()
        if self.path and self.path.exists():
            try:
                self.path.unlink()
            except OSError:
                self.save()

    def align_statements(
        self,
        *,
        statements: List[str],
        context: List[str],
        source_text: str,
    ) -> tuple[List[str], List[dict]]:
        decisions: List[dict] = []
        if not statements:
            return statements, decisions

        known_cards = self.cards_from_entries()
        context_cards = self.card_builder.build(
            context,
            source_text="",
            origin="context",
        )
        current_cards = self.card_builder.build(
            statements,
            source_text=source_text,
            origin="current",
        )

        for card in context_cards:
            self.ensure_card(
                card,
                decisions=decisions,
                action="registry_context_observed",
            )
        for card in current_cards:
            self.ensure_card(
                card,
                decisions=decisions,
                action="registry_predicate_observed",
            )

        if self.mapping_engine:
            discovered, mapping_decisions = self.mapping_engine.discover(
                current_cards=current_cards,
                known_cards=self._dedupe_cards(known_cards + context_cards),
                card_store=self.card_store,
                # Approved and rejected semantic decisions are both cached.
                # Only transient provider failures remain eligible for retry.
                known_mapping_keys=set(self.mappings),
            )
            decisions.extend(mapping_decisions)
            for mapping in discovered:
                self.ensure_validated_mapping(mapping)

        self.save()
        return statements, self.dedupe_decisions(decisions)

    def build_validated_bridges(
        self,
        statements: List[str],
        context: List[str],
    ) -> tuple[List[str], List[dict]]:
        sources = self.schema_aligner.collect_bridge_source_signatures(
            statements,
            "current",
        )
        sources.extend(
            self.schema_aligner.collect_bridge_source_signatures(
                context,
                "context",
            )
        )
        targets = self.schema_aligner.collect_premise_signatures(
            statements,
            origin="current",
        )
        targets.extend(
            self.schema_aligner.collect_premise_signatures(
                context,
                origin="context",
            )
        )
        negated = self.schema_aligner.collect_negated_fact_signatures(
            statements + context
        )

        bridges: List[str] = []
        decisions: List[dict] = []
        seen: set[tuple[str, str, int]] = set()
        for source in sources:
            for target in targets:
                if source["origin"] == "context" and target["origin"] == "context":
                    continue
                if source["head"] == target["head"]:
                    continue
                mapping_result = self._mapping_for_signatures(source, target)
                if not mapping_result:
                    continue
                mapping, bridge_source, bridge_target = mapping_result
                key = (
                    source["head"],
                    target["head"],
                    source["arity"],
                )
                if key in seen:
                    continue
                seen.add(key)

                if self.schema_aligner.bridge_conflicts_with_negated_fact(
                    target,
                    negated,
                ):
                    decisions.append(
                        {
                            "action": "bridge_rejected",
                            "reason": "explicit_negation_conflict",
                            "source": self.schema_aligner.signature_atom(source),
                            "target": self.schema_aligner.signature_atom(target),
                            "mapping_relation": mapping.relation,
                            "mapping_confidence": mapping.confidence,
                            "proof_safe": False,
                        }
                    )
                    continue

                statement = self._bridge_statement(source, target, mapping)
                bridges.append(statement)
                decisions.append(
                    {
                        "action": "bridge_added",
                        "source": self.schema_aligner.signature_atom(source),
                        "target": self.schema_aligner.signature_atom(target),
                        "source_origin": source["origin"],
                        "target_origin": target["origin"],
                        "mapping_relation": mapping.relation,
                        "mapping_confidence": mapping.confidence,
                        "classifier": mapping.classifier,
                        "proof_safe": True,
                        "statement": statement,
                    }
                )
                if len(bridges) >= self.schema_aligner.BRIDGE_MAX_PER_CHUNK:
                    return bridges, decisions
        return bridges, decisions

    def mapping_for_direction(
        self,
        source: str,
        target: str,
        arity: int,
    ) -> PredicateMapping | None:
        mapping_result = self._mapping_for_signatures(
            {"head": source, "arity": arity},
            {"head": target, "arity": arity},
        )
        return mapping_result[0] if mapping_result else None

    def _mapping_for_signatures(
        self,
        source: dict,
        target: dict,
    ) -> tuple[PredicateMapping, dict, dict] | None:
        for key in (
            self.mapping_key(source["head"], target["head"], source["arity"]),
            self.mapping_key(target["head"], source["head"], source["arity"]),
        ):
            mapping = self.mappings.get(key)
            if not mapping or not mapping.proof_safe or mapping.status != "approved":
                continue
            if mapping_allows_direction(
                mapping_source=mapping.source,
                mapping_target=mapping.target,
                relation=mapping.relation,
                requested_source=source["head"],
                requested_target=target["head"],
            ):
                return mapping, source, target
            return None

    def best_relation(self, source: str, target: str, arity: int) -> str | None:
        mapping = self.mapping_for_direction(source, target, arity)
        return mapping.relation if mapping else None

    def validate_bridges(
        self,
        bridges: List[str],
        decisions: List[dict],
    ) -> tuple[List[str], List[dict]]:
        """Compatibility wrapper; new bridges are already gate-approved."""
        allowed = {
            str(decision.get("statement", ""))
            for decision in decisions
            if decision.get("action") == "bridge_added"
            and decision.get("proof_safe") is True
        }
        return [bridge for bridge in bridges if bridge in allowed], decisions

    def classify_mapping(self, left: dict, right: dict) -> dict | None:
        """Lexical diagnostic only; semantic mappings come from the classifier."""
        left_terms = self.schema_aligner.normalized_head_terms(left["head"])
        right_terms = self.schema_aligner.normalized_head_terms(right["head"])
        if left["arity"] != right["arity"] or not left_terms or not right_terms:
            return None
        if left_terms == right_terms:
            return {
                "relation": "exactMatch",
                "confidence": 1.0,
                "reason": "same_normalized_predicate_terms_without_semantic_aliases",
            }
        overlap = left_terms & right_terms
        if not overlap:
            return None
        return {
            "relation": "related",
            "confidence": round(len(overlap) / len(left_terms | right_terms), 3),
            "reason": "lexical_overlap_only",
        }

    def find_exact_existing(
        self,
        signature: dict,
        known_before: dict[str, PredicateEntry],
        known_context: dict[str, PredicateEntry],
    ) -> PredicateEntry | None:
        terms = set(self.schema_aligner.normalized_head_terms(signature["head"]))
        if not terms:
            return None
        for entry in {**known_before, **known_context}.values():
            if entry.predicate == signature["head"] or entry.arity != signature["arity"]:
                continue
            if set(entry.terms) == terms:
                return entry
        return None

    def ensure_card(
        self,
        card: PredicateCard,
        *,
        decisions: List[dict],
        action: str,
    ) -> PredicateEntry:
        key = self.entry_key(card.predicate, card.arity)
        entry = self.entries.get(key)
        if entry is None:
            entry = PredicateEntry(
                predicate=card.predicate,
                arity=card.arity,
                terms=sorted(
                    self.schema_aligner.normalized_head_terms(card.predicate)
                ),
                labels=card.labels or [humanize_predicate(card.predicate)],
                examples=list(card.examples),
                argument_types=list(card.argument_types),
                definitions=[card.definition] if card.definition else [],
                source_atoms=list(card.source_atoms),
                roles=list(card.roles),
                status="observed",
                source_count=0,
            )
            self.entries[key] = entry
            decisions.append(
                {
                    "action": action,
                    "predicate": entry.predicate,
                    "arity": entry.arity,
                    "argument_types": entry.argument_types,
                    "roles": entry.roles,
                    "status": entry.status,
                }
            )
        self._merge_entry(entry, card)
        entry.source_count += 1
        return entry

    def ensure_validated_mapping(
        self,
        validated: ValidatedMapping,
    ) -> PredicateMapping:
        mapping = PredicateMapping(
            source=validated.source.predicate,
            target=validated.target.predicate,
            arity=validated.source.arity,
            relation=validated.relation,
            confidence=validated.confidence,
            reason=validated.reason,
            proof_safe=validated.proof_safe,
            status=validated.status,
            classifier=validated.classifier,
            source_argument_types=validated.source.argument_types,
            target_argument_types=validated.target.argument_types,
            evidence=validated.evidence,
        )
        key = self.mapping_key(mapping.source, mapping.target, mapping.arity)
        existing = self.mappings.get(key)
        if existing:
            if existing.proof_safe and not mapping.proof_safe:
                return existing
            if mapping.proof_safe and not existing.proof_safe:
                self.mappings[key] = mapping
                return mapping
            if existing.confidence > mapping.confidence:
                return existing
        self.mappings[key] = mapping
        return mapping

    def ensure_mapping(
        self,
        *,
        source: str,
        target: str,
        arity: int,
        relation: str,
        confidence: float,
        reason: str,
        proof_safe: bool | None = None,
        status: str | None = None,
    ) -> PredicateMapping:
        """Compatibility helper for callers and tests."""
        safe = relation in PROOF_SAFE_RELATIONS if proof_safe is None else proof_safe
        mapping = PredicateMapping(
            source=source,
            target=target,
            arity=arity,
            relation=relation,
            confidence=confidence,
            reason=reason,
            proof_safe=safe,
            status=status or ("approved" if safe else "candidate"),
        )
        key = self.mapping_key(source, target, arity)
        existing = self.mappings.get(key)
        if existing and existing.confidence >= confidence:
            return existing
        self.mappings[key] = mapping
        return mapping

    def cards_from_entries(self) -> List[PredicateCard]:
        cards: List[PredicateCard] = []
        for entry in self.entries.values():
            cards.append(
                PredicateCard(
                    predicate=entry.predicate,
                    arity=entry.arity,
                    argument_types=entry.argument_types or ["entity"] * entry.arity,
                    definition=(entry.definitions[0] if entry.definitions else ""),
                    labels=entry.labels,
                    examples=entry.examples,
                    source_atoms=entry.source_atoms,
                    roles=entry.roles,
                    origin="registry",
                )
            )
        return cards

    def card_for(self, predicate: str, arity: int) -> PredicateCard | None:
        """Look up one observed predicate card, or None if it was never seen."""
        entry = self.entries.get(f"{predicate}/{int(arity)}")
        if not entry:
            return None
        return PredicateCard(
            predicate=entry.predicate,
            arity=entry.arity,
            argument_types=entry.argument_types or ["entity"] * entry.arity,
            definition=(entry.definitions[0] if entry.definitions else ""),
            labels=entry.labels,
            examples=entry.examples,
            source_atoms=entry.source_atoms,
            roles=entry.roles,
            origin="registry",
        )

    def all_signatures(self, statements: List[str]) -> List[dict]:
        facts, conclusions = self.schema_aligner.collect_available_signatures(
            statements,
            [],
        )
        premises = self.schema_aligner.collect_premise_signatures(
            statements,
            origin="registry",
        )
        negated = self.schema_aligner.collect_negated_fact_signatures(statements)
        signatures = facts + conclusions + premises + negated
        return [
            signature
            for signature in self.dedupe_signatures(signatures)
            if signature["head"] not in self.schema_aligner.skip_heads
        ]

    def rewrite_heads(self, text: str, rewrite_map: dict[str, str]) -> str:
        rewritten = text
        for old, new in sorted(rewrite_map.items(), key=lambda item: -len(item[0])):
            rewritten = re.sub(
                rf"\({re.escape(old)}(?=\s|\))",
                f"({new}",
                rewritten,
            )
        return rewritten

    def dedupe_signatures(self, signatures: Iterable[dict]) -> List[dict]:
        seen: set[tuple] = set()
        result: List[dict] = []
        for signature in signatures:
            key = (
                signature.get("head"),
                tuple(signature.get("args", [])),
                signature.get("arity"),
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(signature)
        return result

    def dedupe_decisions(self, decisions: Iterable[dict]) -> List[dict]:
        seen: set[str] = set()
        result: List[dict] = []
        for decision in decisions:
            key = json.dumps(decision, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            result.append(decision)
        return result

    def label_from_head(self, head: str) -> str:
        return humanize_predicate(head)

    def entry_key(self, predicate: str, arity: int) -> str:
        return f"{predicate}/{arity}"

    def mapping_key(self, source: str, target: str, arity: int) -> tuple[str, str, int]:
        return (source, target, int(arity))

    def _merge_entry(self, entry: PredicateEntry, card: PredicateCard) -> None:
        for value in card.labels:
            if value not in entry.labels:
                entry.labels.append(value)
        for value in card.examples:
            if value not in entry.examples:
                entry.examples.append(value)
        entry.examples = entry.examples[-8:]
        for value in card.source_atoms:
            if value not in entry.source_atoms:
                entry.source_atoms.append(value)
        entry.source_atoms = entry.source_atoms[-12:]
        for value in card.roles:
            if value not in entry.roles:
                entry.roles.append(value)
        if card.definition and card.definition not in entry.definitions:
            entry.definitions.append(card.definition)
        entry.definitions = entry.definitions[-4:]
        if not entry.argument_types:
            entry.argument_types = list(card.argument_types)
        elif argument_types_compatible(entry.argument_types, card.argument_types):
            entry.argument_types = [
                right if left == "entity" else left
                for left, right in zip(entry.argument_types, card.argument_types)
            ]

    def _bridge_statement(
        self,
        source: dict,
        target: dict,
        mapping: PredicateMapping,
    ) -> str:
        variables = self.schema_aligner.bridge_variables(source["arity"])
        source_atom = f"({source['head']} {' '.join(variables)})"
        target_atom = f"({target['head']} {' '.join(variables)})"
        name = self.schema_aligner.bridge_name(source["head"], target["head"])
        strength = max(0.01, min(0.99, mapping.confidence))
        return (
            f"(: {name} "
            f"(Implication (Premises {source_atom}) "
            f"(Conclusions {target_atom})) "
            f"(STV {strength:.2f} 0.80))"
        )

    def _dedupe_cards(self, cards: Iterable[PredicateCard]) -> List[PredicateCard]:
        result: dict[str, PredicateCard] = {}
        for card in cards:
            result.setdefault(card.key, card)
        return list(result.values())
