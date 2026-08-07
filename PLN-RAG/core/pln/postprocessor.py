from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from core.pln.constraint_normalizer import PLNConstraintNormalizer
from core.pln.schema_alignment import PLNSchemaAligner
from core.pln.predicate_registry import PredicateRegistry
from core.pln.symbol_normalization import (
    canonical_symbol as shared_canonical_symbol,
    singularize as shared_singularize,
)


@dataclass
class PLNPostprocessResult:
    statements: List[str] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    alignment_decisions: List[dict] = field(default_factory=list)
    registry_decisions: List[dict] = field(default_factory=list)


class PLNPostprocessor:
    """
    Shared PLN cleanup and query-planning layer.

    Parsers should produce candidate PLN. This class makes those candidates
    safer and more reasoner-aligned before they reach PeTTaChainer.
    """

    STOPWORDS = {
        "a","an","and","are","as","at","be","by","for","from","how","in","is","it","of","on","or","that","the","this","to","was","were","what","when","where","who","why","with","does","do","did","can","could","would","should","has","have","had","if","then","than","into","about","after","before","under","over","not","no","yes",
    }
    STRUCTURAL_HEADS = {
        "Implication",
        "Premises",
        "Conclusions",
        "STV",
        "And",
        "Or",
        "Not",
        "IsA",
        "PointMass",
        "ParticleFromNormal",
        "ParticleFromPairs",
        "GreaterThan",
        "MapDist",
        "Map2Dist",
        "AverageDist",
        "FoldAll",
        "FoldAllValue",
        "Compute",
    }
    PREDICATE_ALIASES = {
        "isa": "IsA",
        "is_a": "IsA",
    }
    GENERIC_SORTALS = {
        "person",
        "people",
        "human",
        "individual",
        "someone",
        "somebody",
        "anyone",
        "anybody",
        "entity",
    }
    QUERY_MARKERS = {"who", "what", "when", "where", "why", "how", "which"}

    def __init__(
        self,
        predicate_registry: PredicateRegistry | None = None,
        *,
        allow_semantic_bridges: bool = False,
    ):
        self._schema_alignment = PLNSchemaAligner(self.STRUCTURAL_HEADS)
        self._constraint_normalizer = PLNConstraintNormalizer(self.STRUCTURAL_HEADS)
        self._predicate_registry = predicate_registry
        self._allow_semantic_bridges = allow_semantic_bridges

    def set_predicate_card_store(self, card_store) -> None:
        if self._predicate_registry:
            self._predicate_registry.set_card_store(card_store)

    def reset_registry(self) -> None:
        if self._predicate_registry:
            self._predicate_registry.clear()

    def process(
        self,
        *,
        text: str,
        statements: List[str],
        queries: List[str],
        context: List[str],
        plan_queries: bool = True,
    ) -> PLNPostprocessResult:
        concepts = self.extract_concepts(self.normalize_text(text))
        protected_constants = self.extract_protected_constants(text)
        proper_name_map = self.extract_proper_name_map(text)
        alignment_decisions: List[dict] = []
        registry_decisions: List[dict] = []

        processed_statements = self.canonicalize_outputs(
            self.dedupe_preserve_order(statements),
            concepts,
            protected_constants,
            proper_name_map,
            context,
        )
        processed_queries = self.canonicalize_outputs(
            self.dedupe_preserve_order(queries),
            concepts,
            protected_constants,
            proper_name_map,
            context,
        )
        property_predicates = self.collect_property_predicate_heads(
            processed_statements + processed_queries + context
        )
        processed_statements = self.normalize_dynamic_property_types(
            processed_statements,
            property_predicates,
        )
        processed_queries = self.normalize_dynamic_property_types(
            processed_queries,
            property_predicates,
        )
        processed_statements = [
            self.repair_missing_universal_variable(stmt)
            for stmt in processed_statements
        ]
        processed_statements, arity_decisions = self.enforce_predicate_arities(
            processed_statements,
            context,
        )
        registry_decisions.extend(arity_decisions)

        if self._predicate_registry and not plan_queries:
            processed_statements, alignment_registry_decisions = (
                self._predicate_registry.align_statements(
                    statements=processed_statements,
                    context=context,
                    source_text=text,
                )
            )
            registry_decisions.extend(alignment_registry_decisions)

        processed_statements = [
            self.prune_generic_sortal_premises(stmt) for stmt in processed_statements
        ]

        processed_statements, constraint_decisions = self._constraint_normalizer.normalize(
            processed_statements,
            source_text=text,
        )
        registry_decisions.extend(constraint_decisions)

        if (
            self._predicate_registry
            and self._allow_semantic_bridges
            and not plan_queries
        ):
            bridges, bridge_decisions = (
                self._predicate_registry.build_validated_bridges(
                    processed_statements,
                    context,
                )
            )
        else:
            bridges, bridge_decisions = [], []
            if self._predicate_registry and not plan_queries:
                bridge_decisions.append(
                    {
                        "action": "semantic_bridges_suppressed",
                        "reason": "proof_authority_disabled",
                        "proof_safe": True,
                    }
                )
        processed_statements.extend(bridges)
        alignment_decisions.extend(registry_decisions)
        alignment_decisions.extend(bridge_decisions)

        processed_statements = self.ensure_statement_format(
            self.filter_statements(processed_statements)
        )
        if plan_queries:
            processed_queries = self.plan_queries(
                question=text,
                queries=processed_queries,
                statements=processed_statements,
                context=context,
            )
        return PLNPostprocessResult(
            statements=processed_statements,
            queries=processed_queries,
            alignment_decisions=alignment_decisions,
            registry_decisions=registry_decisions,
        )

    def ensure_statement_format(self, statements: List[str]) -> List[str]:
        formatted: List[str] = []
        for index, statement in enumerate(statements, start=1):
            clean = " ".join(str(statement).split())
            if not clean:
                continue
            if clean.startswith("(:"):
                query_shaped = re.fullmatch(
                    r"\(:\s+[$?]prf\s+(.+)\s+[$?]tv\)",
                    clean,
                )
                if query_shaped:
                    payload = query_shaped.group(1)
                    name = self.statement_name_from_payload(payload, index)
                    formatted.append(f"(: {name} {payload} (STV 1.0 1.0))")
                    continue
                formatted.append(clean)
                continue
            if re.fullmatch(r"\([A-Za-z][A-Za-z0-9_]*(?:\s+[^()\s]+)+\)", clean):
                name = self.statement_name_from_atom(clean, index)
                formatted.append(f"(: {name} {clean} (STV 1.0 1.0))")
        return self.dedupe_preserve_order(formatted)

    def statement_name_from_payload(self, payload: str, index: int) -> str:
        simple = self._schema_alignment.parse_simple_atom(payload)
        if simple:
            atom = f"({simple['head']} {' '.join(simple['args'])})"
            return self.statement_name_from_atom(atom, index)
        head_match = re.search(r"\(([A-Za-z][A-Za-z0-9_]*)", payload)
        if head_match:
            head = self.canonical_symbol(head_match.group(1), lemmatize=False)
            return f"{head}_rule_{index}"[:80]
        return f"stmt_{index}"

    def statement_name_from_atom(self, atom: str, index: int) -> str:
        match = re.fullmatch(r"\(([A-Za-z][A-Za-z0-9_]*)(?:\s+([^()]+))\)", atom)
        if not match:
            return f"stmt_{index}"
        head = self.canonical_symbol(match.group(1), lemmatize=False)
        args = [
            self.canonical_symbol(arg.lstrip("$?"), lemmatize=False)
            for arg in match.group(2).split()[:2]
            if arg
        ]
        parts = [part for part in args + [head, "fact"] if part]
        return "_".join(parts)[:80] or f"stmt_{index}"

    def normalize_text(self, text: str) -> str:
        text = text.lower().replace("-", " ")
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return " ".join(text.split())

    def extract_concepts(self, normalized_text: str, max_items: int = 12) -> List[str]:
        concepts: List[str] = []
        for token in normalized_text.split():
            if len(token) < 3 or token in self.STOPWORDS:
                continue
            canonical = self.singularize(token)
            if canonical not in concepts:
                concepts.append(canonical)
            if len(concepts) >= max_items:
                break
        return concepts

    def singularize(self, word: str) -> str:
        return shared_singularize(word)


    def repair_missing_universal_variable(self, statement: str) -> str:
        """
        Repair generic person rules where extraction dropped the universal variable.

        Example:
          (Premises (IsA person) (Obese)) (Conclusions (AtRisk))
          -> (Premises (IsA $x person) (Obese $x)) (Conclusions (AtRisk $x))

        This is intentionally narrow: it only fires for implication rules that
        contain the malformed generic sortal marker `(IsA person)`.
        """
        clean = " ".join(str(statement).split())
        if "Implication" not in clean:
            return statement
        if not re.search(r"\(IsA\s+(person|people|human|individual)\)", clean):
            return statement

        repaired = re.sub(
            r"\(IsA\s+(person|people|human|individual)\)",
            r"(IsA $x \1)",
            clean,
        )

        def add_variable(match: re.Match[str]) -> str:
            head = match.group(1)
            if head in self.STRUCTURAL_HEADS:
                return match.group(0)
            return f"({head} $x)"

        return re.sub(r"\(([A-Z][A-Za-z0-9_]*)\)", add_variable, repaired)

    def enforce_predicate_arities(
        self,
        statements: List[str],
        context: List[str],
    ) -> tuple[List[str], List[dict]]:
        """Reject uses that conflict with an established predicate schema."""
        expected: dict[str, set[int]] = {}

        def collect(items: List[str], include_rules: bool = True) -> List[dict]:
            signatures: List[dict] = []
            for item in items:
                signatures.extend(self._schema_alignment.extract_fact_signatures(item))
                signatures.extend(self._schema_alignment.extract_negated_fact_signatures(item))
                if include_rules:
                    signatures.extend(self._schema_alignment.extract_conclusion_signatures(item))
                    signatures.extend(
                        self._schema_alignment.collect_premise_signatures(
                            [item], origin="arity_validation"
                        )
                    )
            return signatures

        for signature in collect(context):
            expected.setdefault(signature["head"], set()).add(signature["arity"])

        rule_signatures: List[dict] = []
        for statement in statements:
            if "Implication" not in statement:
                continue
            rule_signatures.extend(self._schema_alignment.extract_conclusion_signatures(statement))
            rule_signatures.extend(
                self._schema_alignment.collect_premise_signatures(
                    [statement], origin="arity_validation"
                )
            )
        for signature in rule_signatures:
            if signature["head"] not in expected:
                expected.setdefault(signature["head"], set()).add(signature["arity"])

        stable_expected = {
            head: next(iter(arities))
            for head, arities in expected.items()
            if len(arities) == 1
        }
        kept: List[str] = []
        decisions: List[dict] = []
        for statement in statements:
            conflicts = []
            for signature in collect([statement]):
                arity = stable_expected.get(signature["head"])
                if arity is not None and signature["arity"] != arity:
                    conflicts.append(
                        {
                            "predicate": signature["head"],
                            "expected_arity": arity,
                            "actual_arity": signature["arity"],
                        }
                    )
            if conflicts:
                decisions.append(
                    {
                        "status": "rejected",
                        "reason": "predicate_arity_conflict",
                        "statement": statement,
                        "conflicts": conflicts,
                    }
                )
                continue
            kept.append(statement)
        return kept, decisions

    def pluralize(self, word: str) -> str:
        if word.endswith("y") and len(word) > 2:
            return word[:-1] + "ies"
        if word.endswith(("s", "x", "z", "ch", "sh")):
            return word + "es"
        return word + "s"

    def extract_context_predicates(
        self,
        context: List[str],
        max_items: int = 12,
    ) -> List[str]:
        predicates: List[str] = []
        for atom in context:
            for candidate in re.findall(r"\(([A-Za-z][A-Za-z0-9_]*)", atom):
                canonical = self.canonical_head(candidate)
                if canonical and canonical not in predicates:
                    predicates.append(canonical)
                if len(predicates) >= max_items:
                    return predicates
        return predicates

    def extract_context_symbol_map(self, context: List[str]) -> dict[str, str]:
        symbol_map: dict[str, str] = {}
        facts, conclusions = self._schema_alignment.collect_available_signatures(
            [],
            context,
        )
        premises = self._schema_alignment.collect_premise_signatures(
            context,
            origin="context",
        )

        for signature in facts + conclusions + premises:
            for arg in signature["args"]:
                if arg.startswith(("$", "?")):
                    continue
                symbol_map.setdefault(arg, arg)
                singular = self.canonical_symbol(arg, lemmatize=True)
                plural = self.pluralize(singular) if singular else ""
                if singular:
                    symbol_map.setdefault(singular, arg)
                if plural:
                    symbol_map.setdefault(plural, arg)
        return symbol_map

    def extract_context_predicate_map(self, context: List[str]) -> dict[str, str]:
        predicate_map: dict[str, str] = {}
        facts, conclusions = self._schema_alignment.collect_available_signatures(
            [],
            context,
        )
        premises = self._schema_alignment.collect_premise_signatures(
            context,
            origin="context",
        )
        for signature in facts + conclusions + premises:
            head = signature["head"]
            if head in self._schema_alignment.skip_heads:
                continue
            normalized = self.canonical_symbol(head, lemmatize=False)
            normalized_lemma = self.canonical_symbol(head, lemmatize=True)
            if normalized:
                predicate_map.setdefault(normalized, head)
            if normalized_lemma:
                predicate_map.setdefault(normalized_lemma, head)
        return predicate_map

    def canonicalize_outputs(
        self,
        items: List[str],
        concepts: List[str],
        protected_constants: set[str],
        proper_name_map: dict[str, str],
        context: List[str],
    ) -> List[str]:
        if not items:
            return items

        concept_map: dict[str, str] = {}
        for concept in concepts:
            concept_map[concept] = concept
            concept_map[self.pluralize(concept)] = concept
        concept_map.update(self.extract_context_symbol_map(context))
        predicate_map = self.extract_context_predicate_map(context)

        canonical_items = [
            self.canonicalize_atom(
                item,
                concept_map,
                protected_constants,
                proper_name_map,
                predicate_map,
            )
            for item in items
        ]
        canonical_items = [self.normalize_isa_classes(item) for item in canonical_items]
        return self.dedupe_preserve_order(canonical_items)

    def canonicalize_atom(
        self,
        atom: str,
        concept_map: dict[str, str],
        protected_constants: set[str],
        proper_name_map: dict[str, str],
        predicate_map: dict[str, str],
    ) -> str:
        result: List[str] = []
        i = 0
        length = len(atom)

        while i < length:
            ch = atom[i]
            if ch in "$?" or ch.isalpha() or ch == "_":
                start = i
                i += 1
                while i < length and (atom[i].isalnum() or atom[i] == "_"):
                    i += 1
                token = atom[start:i]
                head = self.is_head_position(atom, start)
                result.append(
                    self.normalize_token(
                        token,
                        head,
                        concept_map,
                        protected_constants,
                        proper_name_map,
                        predicate_map,
                    )
                )
                continue
            result.append(ch)
            i += 1

        return "".join(result)

    def is_head_position(self, text: str, index: int) -> bool:
        j = index - 1
        while j >= 0 and text[j].isspace():
            j -= 1
        return j >= 0 and text[j] == "("

    def normalize_token(
        self,
        token: str,
        head: bool,
        concept_map: dict[str, str],
        protected_constants: set[str],
        proper_name_map: dict[str, str],
        predicate_map: dict[str, str],
    ) -> str:
        if token.startswith("$"):
            return "$" + self.canonical_symbol(token[1:], lemmatize=False)
        if token.startswith("?"):
            return "?" + self.canonical_symbol(token[1:], lemmatize=False)

        if head:
            canonical = self.canonical_head(token, predicate_map)
            return canonical if canonical else token

        lowered = token.lower()
        if lowered in proper_name_map:
            return proper_name_map[lowered]
        if lowered in concept_map:
            return concept_map[lowered]
        return self.canonical_symbol(token, protect=lowered in protected_constants)

    def canonical_head(
        self,
        token: str,
        predicate_map: dict[str, str] | None = None,
    ) -> str:
        if token in self.STRUCTURAL_HEADS:
            return token
        normalized = self.canonical_symbol(token, lemmatize=False)
        if predicate_map and normalized in predicate_map:
            return predicate_map[normalized]
        return self.PREDICATE_ALIASES.get(
            normalized,
            token,
        )

    def canonical_symbol(
        self,
        token: str,
        lemmatize: bool = True,
        protect: bool = False,
    ) -> str:
        return shared_canonical_symbol(token, lemmatize=lemmatize, protect=protect)

    def extract_protected_constants(self, text: str) -> set[str]:
        protected: set[str] = set()
        for token in re.findall(r"\b[A-Z][A-Za-z0-9_-]*\b", text):
            canonical = self.canonical_symbol(token, lemmatize=False)
            if canonical:
                protected.add(canonical)
        return protected

    def extract_proper_name_map(self, text: str) -> dict[str, str]:
        proper_names: dict[str, str] = {}
        for token in re.findall(r"\b[A-Z][A-Za-z0-9_-]*\b", text):
            canonical = self.canonical_symbol(token, lemmatize=False)
            if not canonical:
                continue
            proper_names[canonical] = canonical
            singular = self.canonical_symbol(token, lemmatize=True)
            if singular and singular != canonical:
                proper_names[singular] = canonical
        return proper_names

    def filter_statements(self, statements: List[str]) -> List[str]:
        filtered: List[str] = []
        for statement in statements:
            if "Implication" in statement and not self.has_valid_implication_shape(
                statement
            ):
                continue
            if "Implication" not in statement and re.search(
                r"\$[A-Za-z_][A-Za-z0-9_]*",
                statement,
            ):
                continue
            filtered.append(statement)
        return filtered

    def prune_generic_sortal_premises(self, statement: str) -> str:
        if "Implication" not in statement:
            return statement

        match = re.search(r"\(Premises\s+((?:\([^()]+\)\s*)+)\)", statement)
        if not match:
            return statement

        premises = [
            atom.group(0) for atom in re.finditer(r"\([^()]+\)", match.group(1))
        ]
        if len(premises) <= 1:
            return statement

        kept: List[str] = []
        for premise in premises:
            parsed = self._schema_alignment.parse_simple_atom(premise)
            if not parsed:
                kept.append(premise)
                continue
            if (
                parsed["head"] == "IsA"
                and len(parsed["args"]) == 2
                and parsed["args"][0].startswith(("$", "?"))
            ):
                klass = parsed["args"][1].lower()
                if klass in self.GENERIC_SORTALS:
                    continue
            kept.append(premise)

        if len(kept) == len(premises) or not kept:
            return statement

        replacement = "(Premises " + " ".join(kept) + ")"
        return statement[: match.start()] + replacement + statement[match.end() :]

    def has_valid_implication_shape(self, statement: str) -> bool:
        if "Implication" not in statement:
            return True
        return "(Premises" in statement and "(Conclusions" in statement

    def normalize_isa_classes(self, text: str) -> str:
        def repl(match: re.Match[str]) -> str:
            subject = match.group(1)
            klass = match.group(2)
            normalized = self.canonical_symbol(klass, lemmatize=True)
            return f"(IsA {subject} {normalized})"

        return re.sub(r"\(IsA\s+([^()\s]+)\s+([^()\s]+)\)", repl, text)

    def collect_property_predicate_heads(self, items: List[str]) -> set[str]:
        heads: set[str] = set()
        for item in items:
            for match in re.finditer(r"\(([A-Za-z][A-Za-z0-9_]*)", str(item)):
                head = match.group(1)
                if head in self.STRUCTURAL_HEADS:
                    continue
                if not re.fullmatch(r"Is[A-Z][A-Za-z0-9_]*", head):
                    continue
                heads.add(head)
        return heads

    def normalize_dynamic_property_types(
        self,
        items: List[str],
        property_predicates: set[str],
    ) -> List[str]:
        if not property_predicates:
            return items

        normalized: List[str] = []
        for item in items:
            clean = str(item)

            def repl(match: re.Match[str]) -> str:
                subject = match.group(1)
                klass = self.canonical_symbol(match.group(2), lemmatize=True)
                candidate = "Is" + self.pascal_symbol(klass)
                if candidate in property_predicates:
                    return f"({candidate} {subject})"
                return match.group(0)

            normalized.append(
                re.sub(r"\(IsA\s+([^()\s]+)\s+([^()\s]+)\)", repl, clean)
            )
        return normalized

    def pascal_symbol(self, symbol: str) -> str:
        return "".join(
            part[:1].upper() + part[1:]
            for part in re.split(r"[^A-Za-z0-9]+", symbol)
            if part
        )

    def plan_queries(
        self,
        question: str,
        queries: List[str],
        statements: List[str],
        context: List[str],
    ) -> List[str]:
        facts, conclusions = self._schema_alignment.collect_available_signatures(
            statements,
            context,
        )
        facts.extend(
            self._schema_alignment.collect_negated_fact_signatures(
                statements + context
            )
        )
        is_yes_no = self.is_yes_no_question(question)
        planned: List[tuple[int, str]] = []

        if not queries:
            if is_yes_no:
                semantic_fallback = self.build_semantic_context_fallbacks(
                    question,
                    facts,
                    conclusions,
                )
                return self.filter_query_candidates(
                    self.dedupe_preserve_order(semantic_fallback)
                )
            return queries

        for query in queries:
            parsed = self._schema_alignment.parse_query_signature(query)
            if not parsed:
                continue
            score = self.score_query_candidate(parsed, facts, conclusions, is_yes_no)
            if score is None:
                continue
            planned.append((score, query))

        if not planned:
            if is_yes_no:
                semantic_fallback = self.build_semantic_context_fallbacks(
                    question,
                    facts,
                    conclusions,
                )
                fallback = self.build_grounded_yes_no_fallbacks(
                    question,
                    queries,
                    facts,
                    conclusions,
                )
                return self.filter_query_candidates(
                    self.dedupe_preserve_order(semantic_fallback + queries + fallback)
                )
            return queries[:1]

        planned.sort(key=lambda item: item[0], reverse=True)
        ordered = [query for _, query in planned]
        if is_yes_no:
            semantic_fallback = self.build_semantic_context_fallbacks(
                question,
                facts,
                conclusions,
            )
            fallback = self.build_grounded_yes_no_fallbacks(
                question,
                queries,
                facts,
                conclusions,
            )
            ordered = semantic_fallback + ordered + fallback
        return self.filter_query_candidates(self.dedupe_preserve_order(ordered))

    def filter_query_candidates(self, queries: List[str]) -> List[str]:
        filtered: List[str] = []
        for query in queries:
            parsed = self._schema_alignment.parse_query_signature(query)
            if not parsed:
                continue
            if parsed["head"] == "IsA" and parsed["arity"] != 2:
                continue
            if parsed["arity"] == 0:
                continue
            filtered.append(query)
        return filtered

    def build_semantic_context_fallbacks(
        self,
        question: str,
        facts: list[dict],
        conclusions: list[dict],
    ) -> List[str]:
        entities = self.extract_question_constants(question)
        if not entities:
            return []

        question_terms = set(self.extract_question_terms(question))
        candidates: List[tuple[int, str]] = []
        for signature in conclusions + facts:
            if not signature["args"]:
                continue
            score = self.semantic_signature_score(signature, question_terms)
            if score <= 0:
                continue
            for entity in entities:
                grounded = self.ground_signature_with_entity(signature, entity)
                if grounded:
                    candidates.append(
                        (score, self._schema_alignment.signature_to_query(grounded))
                    )

        candidates.sort(key=lambda item: item[0], reverse=True)
        return self.dedupe_preserve_order([query for _, query in candidates])

    def extract_question_terms(self, question: str) -> List[str]:
        terms: List[str] = []
        for token in re.findall(r"\b[A-Za-z0-9][A-Za-z0-9_-]*\b", question):
            lowered = token.lower()
            if len(lowered) < 2 or lowered in self.STOPWORDS:
                continue
            canonical = self.canonical_symbol(lowered, lemmatize=True)
            if canonical and canonical not in terms:
                terms.append(canonical)
        return terms

    def semantic_signature_score(self, signature: dict, question_terms: set[str]) -> int:
        head_terms = self._schema_alignment.normalized_head_terms(signature["head"])
        if not head_terms or not question_terms:
            return 0
        normalized_question = self._schema_alignment.normalized_terms(question_terms)
        overlap = head_terms.intersection(normalized_question)
        domain_overlap = self._schema_alignment.bridge_domain_terms(overlap)
        if not domain_overlap and self._schema_alignment.bridge_domain_terms(head_terms):
            return 0
        return len(overlap) * 4 + len(domain_overlap) * 3

    def ground_signature_with_entity(self, signature: dict, entity: str) -> dict | None:
        args = list(signature["args"])
        if not args:
            return None
        variables = signature.get("variables", [])
        if variables:
            first_var = variables[0]
            args = [entity if arg == first_var else arg for arg in args]
        elif entity not in args:
            if len(args) == 1:
                args = [entity]
            else:
                return None
        return {
            "head": signature["head"],
            "args": args,
            "arity": len(args),
            "variables": [arg for arg in args if arg.startswith(("$", "?"))],
        }

    def build_grounded_yes_no_fallbacks(
        self,
        question: str,
        queries: List[str],
        facts: list[dict],
        conclusions: list[dict],
    ) -> List[str]:
        parsed_queries = [
            parsed
            for query in queries
            if (
                parsed := self._schema_alignment.parse_query_signature(query)
            ) is not None
        ]
        if not parsed_queries:
            return []

        question_symbols = set(self.extract_question_constants(question))
        grounded: List[tuple[int, str]] = []

        for query in parsed_queries:
            if not query["variables"]:
                continue
            for signature in facts + conclusions:
                if signature["variables"]:
                    continue
                if not self.same_shape(query, signature):
                    continue
                if not self.signature_can_bind(query, signature):
                    continue
                if not self.preserves_grounded_args(query, signature, question_symbols):
                    continue

                score = 0
                if question_symbols.intersection(signature["args"]):
                    score += 5
                if signature in facts:
                    score += 3
                if signature in conclusions:
                    score += 2
                grounded.append(
                    (score, self._schema_alignment.signature_to_query(signature))
                )

        grounded.sort(key=lambda item: item[0], reverse=True)
        return self.dedupe_preserve_order([query for _, query in grounded])

    def extract_question_constants(self, question: str) -> List[str]:
        constants: List[str] = []
        for token in re.findall(r"\b[A-Z][A-Za-z0-9_-]*\b", question):
            if token.lower() in {
                "is",
                "are",
                "was",
                "were",
                "does",
                "do",
                "did",
                "can",
                "could",
                "has",
                "have",
                "had",
            }:
                continue
            canonical = self.canonical_symbol(token, lemmatize=False)
            if canonical and canonical not in constants:
                constants.append(canonical)
        return constants



    def score_query_candidate(
        self,
        query: dict,
        facts: list[dict],
        conclusions: list[dict],
        is_yes_no: bool,
    ) -> int | None:
        matching_facts = [sig for sig in facts if self.same_shape(query, sig)]
        matching_conclusions = [
            sig for sig in conclusions if self.same_shape(query, sig)
        ]

        if is_yes_no and query["variables"]:
            if not self.has_witness_path(query, matching_facts, matching_conclusions):
                return None

        score = 0
        if matching_facts:
            score += 6
        if matching_conclusions:
            score += 4
        if not query["variables"]:
            score += 3 if is_yes_no else 1
        else:
            score += 3 if not is_yes_no else 0
        if self.is_fully_grounded_from_signature(query, matching_facts):
            score += 2
        return score if score > 0 else None

    def same_shape(self, left: dict, right: dict) -> bool:
        return left["head"] == right["head"] and left["arity"] == right["arity"]

    def has_witness_path(
        self,
        query: dict,
        matching_facts: list[dict],
        matching_conclusions: list[dict],
    ) -> bool:
        if not query["variables"]:
            return True
        for signature in matching_facts + matching_conclusions:
            if self.signature_can_bind(query, signature):
                return True
        return False

    def signature_can_bind(self, query: dict, signature: dict) -> bool:
        saw_witness = False
        for q_arg, s_arg in zip(query["args"], signature["args"]):
            if q_arg.startswith(("$", "?")):
                if not s_arg.startswith(("$", "?")):
                    saw_witness = True
                continue
            if q_arg != s_arg:
                return False
        return saw_witness or not query["variables"]

    def preserves_grounded_args(
        self,
        query: dict,
        signature: dict,
        question_symbols: set[str],
    ) -> bool:
        for q_arg, s_arg in zip(query["args"], signature["args"]):
            if q_arg.startswith(("$", "?")):
                continue
            if q_arg != s_arg:
                return False
        grounded_question_symbols = {
            arg
            for arg in query["args"]
            if not arg.startswith(("$", "?")) and arg in question_symbols
        }
        return grounded_question_symbols.issubset(set(signature["args"]))

    def is_fully_grounded_from_signature(
        self,
        query: dict,
        matching_facts: list[dict],
    ) -> bool:
        for signature in matching_facts:
            if signature["args"] == query["args"]:
                return True
        return False

    def is_yes_no_question(self, question: str) -> bool:
        tokens = self.normalize_text(question).split()
        return bool(tokens) and tokens[0] in {
            "is",
            "are",
            "was",
            "were",
            "does",
            "do",
            "did",
            "can",
            "could",
            "may",
            "might",
            "must",
            "shall",
            "should",
            "has",
            "have",
            "had",
        }

    def dedupe_preserve_order(self, items: List[str]) -> List[str]:
        seen = set()
        deduped: List[str] = []
        for item in items:
            clean = " ".join(item.split())
            if clean and clean not in seen:
                seen.add(clean)
                deduped.append(clean)
        return deduped
