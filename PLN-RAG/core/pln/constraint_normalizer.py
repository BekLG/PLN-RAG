from __future__ import annotations

import re
from typing import Iterable, List

from core.pln.schema_alignment import PLNSchemaAligner
from core.pln.semantic_ir import (
    Literal,
    Measurement,
    NumericConstraint,
    NumericValue,
)
from core.pln.symbol_normalization import canonical_symbol


NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

COMPARATOR_TERMS = {
    "above",
    "at",
    "least",
    "below",
    "under",
    "less",
    "more",
    "than",
    "over",
    "greater",
    "exceed",
    "exceeds",
    "minimum",
    "maximum",
}
EVENT_TERMS = {
    "has",
    "have",
    "is",
    "are",
    "was",
    "were",
    "stays",
    "stay",
    "reached",
    "reach",
    "measured",
    "maintained",
    "been",
    "completed",
}
FREQUENCY_TERMS = {
    "frequent",
    "frequently",
    "time",
    "times",
    "week",
    "daily",
    "day",
    "every",
    "almost",
}
NEGATIVE_EVIDENCE_GENERIC_TERMS = {
    "after",
    "before",
    "day",
    "days",
    "for",
    "next",
    "no",
    "not",
    "two",
    "within",
}


class PLNConstraintNormalizer:
    """Add deterministic support facts for numeric, frequency, and role constraints."""

    def __init__(self, structural_heads: Iterable[str]):
        self._aligner = PLNSchemaAligner(structural_heads)

    def normalize(
        self,
        statements: List[str],
        source_text: str = "",
    ) -> tuple[List[str], List[dict]]:
        existing = {self._statement_body(statement) for statement in statements}
        facts = self._facts(statements)
        additions: List[str] = []
        decisions: List[dict] = []

        morphology_atoms = self._morphological_derivations(
            statements,
            facts,
            existing,
        )
        for atom, reason in morphology_atoms:
            statement = self._statement(atom, reason)
            additions.append(statement)
            existing.add(atom)
            decisions.append(
                {"status": "added", "reason": reason, "statement": statement}
            )

        facts = self._facts(statements + additions)
        numeric_atoms = self._numeric_derivations(
            statements + additions,
            facts,
            existing,
            source_text,
        )
        frequency_atoms = self._frequency_derivations(
            statements + additions,
            facts,
            existing,
        )
        for atom, reason in numeric_atoms + frequency_atoms:
            statement = self._statement(atom, reason)
            additions.append(statement)
            existing.add(atom)
            decisions.append(
                {"status": "added", "reason": reason, "statement": statement}
            )

        facts = self._facts(statements + additions)
        negative_atoms = self._negative_derivations(
            statements + additions,
            existing,
        )
        source_negative_atoms = self._source_text_negative_derivations(
            statements + additions,
            facts,
            existing,
            source_text,
        )
        for atom, reason in negative_atoms + source_negative_atoms:
            statement = self._statement(atom, reason)
            additions.append(statement)
            existing.add(atom)
            decisions.append(
                {"status": "added", "reason": reason, "statement": statement}
            )

        facts = self._facts(statements + additions)
        type_atoms = self._role_type_derivations(statements, facts, existing)
        for atom, reason in type_atoms:
            statement = self._statement(atom, reason)
            additions.append(statement)
            existing.add(atom)
            decisions.append(
                {"status": "added", "reason": reason, "statement": statement}
            )

        return statements + additions, decisions

    def _numeric_derivations(
        self,
        statements: List[str],
        facts: List[dict],
        existing: set[str],
        source_text: str,
    ) -> List[tuple[str, str]]:
        results: List[tuple[str, str]] = []
        measurements = self._measurements(facts)
        premises = self._premises(statements)
        for target in premises:
            constraint = self._numeric_constraint(target, source_text)
            if not constraint:
                continue
            entity_hints = self._candidate_entities_for_target(
                target,
                statements,
                facts,
            )
            for measurement in measurements:
                entity = measurement.entity
                if measurement.owner_missing:
                    if len(entity_hints) != 1:
                        continue
                    entity = next(iter(entity_hints))
                if not entity or entity.startswith(("$", "?")):
                    continue
                if entity_hints and entity not in entity_hints:
                    continue
                if not self._property_terms_compatible(
                    constraint.property_terms,
                    measurement.property_terms,
                ):
                    continue
                if not self._compatible_units(
                    constraint.threshold.unit,
                    measurement.numeric.unit,
                ):
                    continue
                if not self._compare(
                    measurement.numeric.value,
                    constraint.threshold.value,
                    constraint.comparator,
                ):
                    continue
                if not self._secondary_constraints_satisfied(
                    constraint,
                    measurements,
                    entity,
                ):
                    continue
                atom = self._ground_target_atom(target, entity)
                if atom in existing:
                    continue
                results.append((atom, "numeric_threshold_satisfied"))
                existing.add(atom)
        results.extend(
            self._variable_numeric_derivations(
                statements,
                measurements,
                facts,
                existing,
            )
        )
        return results

    def _source_text_negative_derivations(
        self,
        statements: List[str],
        facts: List[dict],
        existing: set[str],
        source_text: str,
    ) -> List[tuple[str, str]]:
        if not source_text:
            return []
        results: List[tuple[str, str]] = []
        sentences = [
            sentence
            for sentence in re.split(r"[.!?]+", source_text)
            if self._has_explicit_negative_marker(sentence)
        ]
        if not sentences:
            return []
        for target in self._negated_premises(statements):
            hints = self._candidate_entities_for_target(target, statements, facts)
            if len(hints) != 1:
                continue
            entity = next(iter(hints))
            atom = self._negated_atom_for_target(target, entity)
            if not atom or atom in existing:
                continue
            if not any(self._sentence_matches_negative_target(sentence, target) for sentence in sentences):
                continue
            results.append((atom, "explicit_negative_supported_by_source_text"))
            existing.add(atom)
        return results

    def _has_explicit_negative_marker(self, sentence: str) -> bool:
        return bool(
            re.search(
                r"\b(no|not|never|without|none|neither|nor)\b",
                sentence.lower(),
            )
        )

    def _sentence_matches_negative_target(self, sentence: str, target: dict) -> bool:
        target_terms = self._literal_terms(target) - NEGATIVE_EVIDENCE_GENERIC_TERMS
        sentence_terms = self._semantic_terms(sentence) - NEGATIVE_EVIDENCE_GENERIC_TERMS
        if not target_terms:
            return False
        required = target_terms
        if len(required) > 2:
            required = {
                term
                for term in required
                if term not in {"within", "day", "two"}
            } or target_terms
        return required.issubset(sentence_terms)

    def _negated_atom_for_target(self, target: dict, entity: str) -> str:
        grounded = [
            entity if arg.startswith(("$", "?")) else arg
            for arg in target["args"]
        ]
        if any(arg.startswith(("$", "?")) for arg in grounded):
            return ""
        return f"(Not ({target['head']} {' '.join(grounded)}))"

    def _variable_numeric_derivations(
        self,
        statements: List[str],
        measurements: List[Measurement],
        facts: List[dict],
        existing: set[str],
    ) -> List[tuple[str, str]]:
        results: List[tuple[str, str]] = []
        for statement in statements:
            if "Implication" not in statement:
                continue
            premises = self._premises([statement]) + self._numeric_comparator_premises(
                statement
            )
            comparators = [
                premise
                for premise in premises
                if self._comparator(premise["head"])
                and premise["args"]
                and self._is_variable(premise["args"][0])
                and self._numeric_values(premise["args"][1:])
            ]
            for comparator in comparators:
                value_var = comparator["args"][0]
                threshold = self._numeric_values(comparator["args"][1:])[0]
                comparison = self._comparator(comparator["head"])
                for target in premises:
                    if target is comparator or self._comparator(target["head"]):
                        continue
                    if value_var not in target["args"]:
                        continue
                    entity_var = next(
                        (
                            arg
                            for arg in target["args"]
                            if self._is_variable(arg) and arg != value_var
                        ),
                        "",
                    )
                    if not entity_var:
                        continue
                    property_terms = self._measurement_terms(target["head"])
                    entity_hints = self._candidate_entities_for_target(
                        target,
                        statements,
                        facts,
                    )
                    for measurement in measurements:
                        entity = measurement.entity
                        if measurement.owner_missing:
                            if len(entity_hints) != 1:
                                continue
                            entity = next(iter(entity_hints))
                        if not entity or self._is_variable(entity):
                            continue
                        if entity_hints and entity not in entity_hints:
                            continue
                        if not self._property_terms_compatible(
                            property_terms,
                            measurement.property_terms,
                        ):
                            continue
                        if not self._compatible_units(
                            threshold.unit,
                            measurement.numeric.unit,
                        ):
                            continue
                        if not self._compare(
                            measurement.numeric.value,
                            threshold.value,
                            comparison,
                        ):
                            continue
                        value_token = self._measurement_value_token(measurement)
                        if not value_token:
                            continue
                        target_atom = self._ground_variable_numeric_atom(
                            target,
                            entity_var,
                            entity,
                            value_var,
                            value_token,
                        )
                        comparator_atom = self._ground_variable_numeric_atom(
                            comparator,
                            entity_var,
                            entity,
                            value_var,
                            value_token,
                        )
                        for atom in (target_atom, comparator_atom):
                            if not atom or atom in existing:
                                continue
                            results.append((atom, "numeric_variable_satisfied"))
                            existing.add(atom)
        return results

    def _numeric_comparator_premises(self, statement: str) -> List[dict]:
        premises: List[dict] = []
        for block in self._aligner.extract_named_blocks(statement, "Premises"):
            for match in re.finditer(
                r"\(([A-Za-z][A-Za-z0-9_]*)((?:\s+[^()\s]+)+)\)",
                block,
            ):
                parsed = self._aligner.parse_simple_atom(match.group(0))
                if not parsed or not self._comparator(parsed["head"]):
                    continue
                tagged = dict(parsed)
                tagged.update(
                    {
                        "origin": "constraint_normalizer",
                        "role": "premise",
                    }
                )
                premises.append(tagged)
        return premises

    def _morphological_derivations(
        self,
        statements: List[str],
        facts: List[dict],
        existing: set[str],
    ) -> List[tuple[str, str]]:
        results: List[tuple[str, str]] = []
        for target in self._premises(statements):
            if self._comparator(target["head"]):
                continue
            target_terms = self._semantic_terms(target["head"])
            if not target_terms:
                continue
            for fact in facts:
                if target["arity"] != fact["arity"]:
                    continue
                if target_terms != self._semantic_terms(fact["head"]):
                    continue
                grounded = self._bind_target_to_fact(target["args"], fact["args"])
                if grounded is None:
                    continue
                atom = f"({target['head']} {' '.join(grounded)})"
                if atom in existing:
                    continue
                results.append((atom, "local_schema_morphology_match"))
                existing.add(atom)
        return results

    def _negative_derivations(
        self,
        statements: List[str],
        existing: set[str],
    ) -> List[tuple[str, str]]:
        results: List[tuple[str, str]] = []
        targets = self._negated_premises(statements)
        sources = self._negated_facts(statements)
        positive_facts = self._facts(statements)
        for target in targets:
            hints = self._candidate_entities_for_target(
                target,
                statements,
                positive_facts,
            )
            target_terms = self._literal_terms(target)
            for source in sources:
                if target["arity"] != source["arity"]:
                    continue
                if target_terms != self._literal_terms(source):
                    continue
                source_args = list(source["args"])
                if source_args and self._is_unattached_entity(source_args[0]):
                    if len(hints) != 1:
                        continue
                    source_args[0] = next(iter(hints))
                grounded = self._bind_target_to_fact(target["args"], source_args)
                if grounded is None:
                    if len(hints) != 1:
                        continue
                    grounded = [
                        next(iter(hints)) if arg.startswith(("$", "?")) else arg
                        for arg in target["args"]
                    ]
                atom = f"(Not ({target['head']} {' '.join(grounded)}))"
                if atom in existing:
                    continue
                results.append((atom, "explicit_negative_local_schema_match"))
                existing.add(atom)
        return results

    def _frequency_derivations(
        self,
        statements: List[str],
        facts: List[dict],
        existing: set[str],
    ) -> List[tuple[str, str]]:
        results: List[tuple[str, str]] = []
        for target in self._premises(statements):
            if not ({"frequent", "frequently"} & self._head_terms(target["head"])):
                continue
            for fact in facts:
                if fact["arity"] != target["arity"] or not fact["args"]:
                    continue
                if not self._is_high_frequency_head(fact["head"]):
                    continue
                if not self._heads_share_domain_action(target["head"], fact["head"]):
                    continue
                binding = self._bind_target_to_fact(target["args"], fact["args"])
                if binding is None:
                    continue
                atom = f"({target['head']} {' '.join(binding)})"
                if atom in existing:
                    continue
                results.append((atom, "frequency_threshold_satisfied"))
                existing.add(atom)
        return results

    def _role_type_derivations(
        self,
        statements: List[str],
        facts: List[dict],
        existing: set[str],
    ) -> List[tuple[str, str]]:
        results: List[tuple[str, str]] = []
        fact_by_head = {}
        for fact in facts:
            fact_by_head.setdefault(fact["head"], []).append(fact)

        for rule in statements:
            premises = self._premises([rule])
            negated_keys = {
                (premise["head"], tuple(premise["args"]))
                for premise in self._negated_premises([rule])
            }
            type_premises = self._type_premises(rule)
            if not type_premises:
                continue
            for type_premise in type_premises:
                variable, klass = type_premise["args"]
                if klass in {"person", "people", "entity"}:
                    continue
                required = [
                    premise
                    for premise in premises
                    if premise["head"] != "IsA" and variable in premise["args"]
                    and (premise["head"], tuple(premise["args"])) not in negated_keys
                ]
                if not required:
                    continue
                candidate_entities: set[str] | None = None
                for premise in required:
                    matches = {
                        entity
                        for fact in fact_by_head.get(premise["head"], [])
                        for entity in [self._entity_for_variable(premise, fact, variable)]
                        if entity
                    }
                    candidate_entities = (
                        matches
                        if candidate_entities is None
                        else candidate_entities & matches
                    )
                    if not candidate_entities:
                        break
                for entity in sorted(candidate_entities or set()):
                    if entity == klass:
                        continue
                    atom = f"(IsA {entity} {klass})"
                    if atom in existing:
                        continue
                    results.append((atom, "role_type_inferred_from_rule_schema"))
                    existing.add(atom)
        return results

    def _type_premises(self, statement: str) -> List[dict]:
        premises: List[dict] = []
        for match in re.finditer(
            r"\(IsA\s+([$?][A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z][A-Za-z0-9_]*)\)",
            statement,
        ):
            premises.append(
                {
                    "head": "IsA",
                    "args": [match.group(1), match.group(2)],
                    "arity": 2,
                    "variables": [match.group(1)],
                }
            )
        return premises

    def _premises(self, statements: List[str]) -> List[dict]:
        results: List[dict] = []
        for statement in statements:
            results.extend(
                self._aligner.collect_premise_signatures(
                    [statement],
                    origin="constraint_normalizer",
                )
            )
        return results

    def _facts(self, statements: List[str]) -> List[dict]:
        facts: List[dict] = []
        for statement in statements:
            facts.extend(self._aligner.extract_fact_signatures(statement))
        return facts

    def _negated_facts(self, statements: List[str]) -> List[dict]:
        facts: List[dict] = []
        for statement in statements:
            facts.extend(self._aligner.extract_negated_fact_signatures(statement))
        return facts

    def _negated_premises(self, statements: List[str]) -> List[dict]:
        premises: List[dict] = []
        for statement in statements:
            if "Implication" not in statement:
                continue
            for match in re.finditer(r"\(Not\s+(\([^()]+\))\)", statement):
                parsed = self._aligner.parse_simple_atom(match.group(1))
                if parsed:
                    premises.append(parsed)
        return premises

    def _comparator(self, head: str) -> str:
        terms = self._head_terms(head)
        if {"at", "least"}.issubset(terms) or "minimum" in terms:
            return ">="
        if (
            {"more", "than"}.issubset(terms)
            or {"greater", "than"}.issubset(terms)
            or "above" in terms
            or "over" in terms
            or "exceed" in terms
            or "exceeds" in terms
        ):
            return ">"
        if "below" in terms or "under" in terms:
            return "<"
        if {"less", "than"}.issubset(terms):
            return "<"
        return ""

    def _numeric_constraint(
        self,
        target: dict,
        source_text: str,
    ) -> NumericConstraint | None:
        comparator = self._comparator(target["head"])
        values = self._numeric_values(target["args"][1:])
        if not values:
            embedded = self._numeric_value(target["head"])
            if embedded is not None:
                values = [NumericValue(embedded)]
        if not values:
            return None
        if not comparator:
            comparator = self._source_comparator(target, source_text)
        if not comparator:
            return None
        return NumericConstraint(
            target=Literal(target["head"], tuple(target["args"])),
            comparator=comparator,
            threshold=values[0],
            property_terms=frozenset(self._measurement_terms(target["head"])),
            secondary=tuple(values[1:]),
        )

    def _measurements(self, facts: List[dict]) -> List[Measurement]:
        measurements: List[Measurement] = []
        for fact in facts:
            args = fact["args"]
            if not args:
                continue
            literal = Literal(fact["head"], tuple(args))
            if fact["head"] == "Has" and len(args) >= 3:
                values = self._numeric_values(args[2:])
                if not values:
                    continue
                property_symbol = args[1]
                entity = args[0]
                owner_missing = canonical_symbol(property_symbol) in {
                    "value",
                    "level",
                    "measurement",
                }
                if owner_missing:
                    property_symbol = args[0]
                    entity = ""
                measurements.append(
                    Measurement(
                        entity=entity,
                        property_terms=frozenset(
                            self._semantic_terms(property_symbol)
                        ),
                        numeric=values[0],
                        source=literal,
                        owner_missing=owner_missing,
                    )
                )
                continue

            values = self._numeric_values(args[1:])
            if not values:
                continue
            measurements.append(
                Measurement(
                    entity=args[0],
                    property_terms=frozenset(
                        self._measurement_terms(fact["head"])
                    ),
                    numeric=values[0],
                    source=literal,
                )
            )
        return measurements

    def _numeric_values(self, args: List[str]) -> List[NumericValue]:
        values: List[NumericValue] = []
        index = 0
        while index < len(args):
            value = self._numeric_value(args[index])
            if value is None:
                index += 1
                continue
            units: List[str] = []
            cursor = index
            inline_unit = self._numeric_unit(args[index])
            if inline_unit:
                units.append(inline_unit)
            cursor += 1
            while cursor < len(args) and self._numeric_value(args[cursor]) is None:
                unit = canonical_symbol(args[cursor])
                if unit:
                    units.append(unit)
                cursor += 1
            values.append(
                NumericValue(value=value, unit=self._normalize_unit("_".join(units)))
            )
            index = cursor
        return values

    def _source_comparator(self, target: dict, source_text: str) -> str:
        if not source_text:
            return ""
        target_terms = self._semantic_terms(target["head"])
        for sentence in re.split(r"[.!?]+", source_text.lower()):
            sentence_terms = self._semantic_terms(sentence)
            if len(target_terms & sentence_terms) < min(2, len(target_terms)):
                continue
            if re.search(r"\b(at least|or more|no fewer than|minimum of)\b", sentence):
                return ">="
            if re.search(r"\b(above|over|more than|greater than|exceeds?)\b", sentence):
                return ">"
            if re.search(r"\b(at most|or fewer|no more than|maximum of)\b", sentence):
                return "<="
            if re.search(r"\b(below|under|less than|fewer than)\b", sentence):
                return "<"
        return ""

    def _candidate_entities_for_target(
        self,
        target: dict,
        statements: List[str],
        facts: List[dict],
    ) -> set[str]:
        variables = set(target.get("variables", []))
        if not variables:
            return {target["args"][0]} if target["args"] else set()
        variable = next(iter(variables))
        candidates: set[str] = set()
        fact_by_head: dict[str, List[dict]] = {}
        for fact in facts:
            fact_by_head.setdefault(fact["head"], []).append(fact)
        for statement in statements:
            if "Implication" not in statement or f"({target['head']} " not in statement:
                continue
            for premise in self._premises([statement]):
                if premise["head"] == target["head"] or variable not in premise["args"]:
                    continue
                for fact in fact_by_head.get(premise["head"], []):
                    entity = self._entity_for_variable(premise, fact, variable)
                    if entity:
                        candidates.add(entity)
        return candidates

    def _secondary_constraints_satisfied(
        self,
        constraint: NumericConstraint,
        measurements: List[Measurement],
        entity: str,
    ) -> bool:
        for required in constraint.secondary:
            found = False
            for measurement in measurements:
                if measurement.entity != entity:
                    continue
                if not self._compatible_units(required.unit, measurement.numeric.unit):
                    continue
                if measurement.numeric.value >= required.value:
                    found = True
                    break
            if not found:
                return False
        return True

    def _first_numeric(self, args: List[str]) -> NumericValue | None:
        for index, arg in enumerate(args):
            value = self._numeric_value(arg)
            if value is None:
                continue
            unit = self._numeric_unit(arg)
            if not unit and index + 1 < len(args):
                next_unit = canonical_symbol(args[index + 1])
                if next_unit and self._numeric_value(next_unit) is None:
                    unit = next_unit
            return NumericValue(value=value, unit=unit)
        return None

    def _numeric_value(self, token: str) -> float | None:
        normalized = canonical_symbol(str(token))
        if normalized in NUMBER_WORDS:
            return float(NUMBER_WORDS[normalized])
        for part in normalized.split("_"):
            if part in NUMBER_WORDS:
                return float(NUMBER_WORDS[part])
        match = re.search(r"\d+(?:\.\d+)?", str(token))
        if not match:
            return None
        return float(match.group(0))

    def _numeric_unit(self, token: str) -> str:
        normalized = canonical_symbol(str(token))
        if normalized in NUMBER_WORDS:
            return ""
        parts = [part for part in normalized.split("_") if part]
        non_numeric = [
            re.sub(r"\d+(?:\.\d+)?", "", part).strip("_")
            for part in parts
            if part not in NUMBER_WORDS
            and not re.fullmatch(r"\d+(?:\.\d+)?", part)
            and re.sub(r"\d+(?:\.\d+)?", "", part).strip("_")
        ]
        compact = re.sub(r"\d+(?:\.\d+)?", "", normalized).strip("_")
        return "_".join(non_numeric) or compact

    def _compare(self, left: float, right: float, comparator: str) -> bool:
        if comparator == ">=":
            return left >= right
        if comparator == ">":
            return left > right
        if comparator == "<":
            return left < right
        if comparator == "<=":
            return left <= right
        return False

    def _compatible_units(self, left: str, right: str) -> bool:
        if not left or not right:
            return True
        return self._normalize_unit(left) == self._normalize_unit(right)

    def _normalize_unit(self, unit: str) -> str:
        normalized = canonical_symbol(unit)
        aliases = {
            "degree_celsius": "celsius",
            "degree_celsiu": "celsius",
            "degrees_celsius": "celsius",
            "minute": "minute",
            "minutes": "minute",
            "percent": "percent",
            "percentage": "percent",
        }
        return aliases.get(normalized, normalized)

    def _property_terms_compatible(
        self,
        target: frozenset[str],
        measured: frozenset[str],
    ) -> bool:
        if not target or not measured:
            return False
        overlap = set(target) & set(measured)
        return bool(overlap) and (
            set(target).issubset(measured)
            or set(measured).issubset(target)
            or len(overlap) >= 2
        )

    def _heads_are_measurement_compatible(self, target: str, fact: str) -> bool:
        target_terms = self._measurement_terms(target)
        fact_terms = self._measurement_terms(fact)
        if not target_terms or not fact_terms:
            return False
        return target_terms.issubset(fact_terms) or fact_terms.issubset(target_terms)

    def _heads_share_domain_action(self, target: str, fact: str) -> bool:
        target_terms = self._action_terms(target)
        fact_terms = self._action_terms(fact)
        return bool(target_terms and fact_terms and target_terms & fact_terms)

    def _measurement_terms(self, head: str) -> set[str]:
        return {
            term
            for term in self._semantic_terms(head)
            if term not in COMPARATOR_TERMS
            and term not in EVENT_TERMS
            and term not in FREQUENCY_TERMS
            and not term.isdigit()
        }

    def _action_terms(self, head: str) -> set[str]:
        return {
            term
            for term in self._head_terms(head)
            if term not in COMPARATOR_TERMS
            and term not in EVENT_TERMS
            and term not in FREQUENCY_TERMS
            and term not in NUMBER_WORDS
        }

    def _head_terms(self, head: str) -> set[str]:
        spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", head)
        spaced = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", "_", spaced)
        terms = {
            canonical_symbol(part)
            for part in re.split(r"[^A-Za-z0-9]+", spaced)
            if part
        }
        expanded = set(terms)
        for term in terms:
            if term.endswith("week") and term != "week":
                expanded.add("week")
            if term.endswith("day") and term != "day":
                expanded.add("day")
        return expanded

    def _semantic_terms(self, value: str) -> set[str]:
        terms = self._head_terms(value)
        normalized: set[str] = set()
        for term in terms:
            if term in {"has", "have", "is", "are", "was", "were", "been"}:
                continue
            if term.endswith("ed") and len(term) > 4:
                term = term[:-2]
            elif term.endswith("ing") and len(term) > 5:
                term = term[:-3]
            normalized.add(term)
        return normalized

    def _literal_terms(self, signature: dict) -> set[str]:
        terms = self._semantic_terms(signature["head"])
        for arg in signature["args"]:
            if arg.startswith(("$", "?")) or canonical_symbol(arg) in {
                "",
                "null",
                "unknown",
                "none",
            }:
                continue
            terms.update(self._semantic_terms(arg))
        return {term for term in terms if term not in {"null"}}

    def _is_unattached_entity(self, value: str) -> bool:
        normalized = canonical_symbol(value)
        return normalized in {"", "null", "unknown", "none"} or normalized in {
            "replacement_route",
        }

    def _is_high_frequency_head(self, head: str) -> bool:
        terms = self._head_terms(head)
        if "frequently" in terms or "frequent" in terms or "daily" in terms:
            return True
        if {"every", "day"}.issubset(terms):
            return True
        if {"time", "week"}.issubset(terms) or {"times", "week"}.issubset(terms):
            return any(
                (NUMBER_WORDS.get(term, -1) >= 3)
                or (term.isdigit() and int(term) >= 3)
                for term in terms
            )
        return False

    def _bind_target_to_fact(
        self,
        target_args: List[str],
        fact_args: List[str],
    ) -> List[str] | None:
        if len(target_args) != len(fact_args):
            return None
        bindings: dict[str, str] = {}
        grounded: List[str] = []
        for target_arg, fact_arg in zip(target_args, fact_args):
            if target_arg.startswith(("$", "?")):
                current = bindings.setdefault(target_arg, fact_arg)
                if current != fact_arg:
                    return None
                grounded.append(fact_arg)
                continue
            if target_arg != fact_arg:
                return None
            grounded.append(target_arg)
        return grounded

    def _is_variable(self, value: str) -> bool:
        return isinstance(value, str) and value.startswith(("$", "?"))

    def _measurement_value_token(self, measurement: Measurement) -> str:
        for arg in measurement.source.args:
            value = self._numeric_value(arg)
            if value is None or value != measurement.numeric.value:
                continue
            if self._compatible_units(
                self._numeric_unit(arg),
                measurement.numeric.unit,
            ):
                return arg
        return ""

    def _ground_variable_numeric_atom(
        self,
        target: dict,
        entity_var: str,
        entity: str,
        value_var: str,
        value_token: str,
    ) -> str:
        grounded: List[str] = []
        for arg in target["args"]:
            if arg == entity_var:
                grounded.append(entity)
            elif arg == value_var:
                grounded.append(value_token)
            elif self._is_variable(arg):
                return ""
            else:
                grounded.append(arg)
        return f"({target['head']} {' '.join(grounded)})"

    def _entity_for_variable(self, premise: dict, fact: dict, variable: str) -> str:
        if premise["arity"] != fact["arity"]:
            return ""
        for premise_arg, fact_arg in zip(premise["args"], fact["args"]):
            if premise_arg == variable and not fact_arg.startswith(("$", "?")):
                return fact_arg
            if not premise_arg.startswith(("$", "?")) and premise_arg != fact_arg:
                return ""
        return ""

    def _ground_target_atom(self, target: dict, entity: str) -> str:
        args = [entity if arg.startswith(("$", "?")) else arg for arg in target["args"]]
        return f"({target['head']} {' '.join(args)})"

    def _statement_body(self, statement: str) -> str:
        match = re.fullmatch(
            r"\(:\s+[^\s]+\s+(\(.+\))\s+\(STV\s+[^\s]+\s+[^\s]+\)\)",
            " ".join(str(statement).split()),
        )
        return match.group(1) if match else ""

    def _statement(self, atom: str, reason: str) -> str:
        parsed = self._aligner.parse_simple_atom(atom)
        if not parsed:
            name = "constraint_derived"
        else:
            parts = [*parsed["args"][:2], parsed["head"], reason]
            name = canonical_symbol("_".join(parts), lemmatize=False)[:80]
        return f"(: {name} {atom} (STV 1.0 1.0))"
