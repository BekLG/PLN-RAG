from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from core.pln.constraint_normalizer import PLNConstraintNormalizer
from core.pln.schema_alignment import PLNSchemaAligner
from core.pln.predicate_registry import PredicateRegistry


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
        processed_statements = self.repair_portion_size_fact_arity(
            processed_statements
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

        # Infer types for proper names that appear in statements
        inferred_types = self.infer_entity_types(processed_statements, proper_name_map)
        processed_statements.extend(inferred_types)

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
        if len(word) <= 3:
            return word
        if word.endswith("ies") and len(word) > 4:
            return word[:-3] + "y"
        if word.endswith("ses") and len(word) > 4:
            return word[:-2]
        if word.endswith("s") and not word.endswith(("ss", "us", "is")):
            return word[:-1]
        return word
    def generate_universal_identity(self, statements: List[str]) -> List[str]:
        """
        Extract all lowercase terms from statements and generate (IsA term term).
        This helps logic reasoners unify self-identity constraints easily.
        """
        import re
        terms = set()
        for stmt in statements:
            s_clean = stmt.replace("(", " ").replace(")", " ")
            for token in s_clean.split():
                if token.islower() and not re.match(r'^[0-9.]+$', token):
                    if token not in {"and", "or", "not", "stv"}:
                        terms.add(token)
        
        return [f"(IsA {t} {t})" for t in terms]
    def infer_entity_types(self, statements, proper_name_map):
        """
        Infer (IsA entity person) for proper names that appear as the FIRST argument
        (subject/actor) of predicates in statements, but have no explicit type declaration.
        Uses proper_name_map from source text as the candidate set.
        Entities that only appear as objects (2nd+ arg) are not inferred as persons.
        """
        candidate_entities = set(proper_name_map.values())
        # Exclude identifiers containing digits (Object-77, Unit-12, Rack-01, etc.)
        non_person = re.compile("[0-9]")
        person_candidates = {
            e for e in candidate_entities
            if len(e) > 1 and not non_person.search(e)
        }

        # Collect entities that already have explicit type declarations
        entities_with_types = set()
        for stmt in statements:
            for match in re.finditer("\\(IsA\\s+(\\S+)\\s+\\S+\\)", stmt):
                entities_with_types.add(match.group(1))

        # Collect entities that appear as the FIRST argument of a predicate
        # Pattern: (PredicateName first_arg ...) where PredicateName starts uppercase
        entities_as_subject = set()
        for stmt in statements:
            for match in re.finditer("\\(([A-Z][A-Za-z0-9_]*)\\s+(\\S+)", stmt):
                pred = match.group(1)
                first_arg = match.group(2)
                # Skip structural keywords and variables
                if pred in self.STRUCTURAL_HEADS:
                    continue
                if first_arg.startswith("$") or first_arg.startswith("?"):
                    continue
                entities_as_subject.add(first_arg)

        inferred = []
        for entity in sorted(person_candidates):
            if entity in entities_with_types:
                continue
            if entity not in entities_as_subject:
                continue
            inferred.append(
                "(: " + entity + "_is_person (IsA " + entity + " person) (STV 1.0 1.0))"
            )
        return inferred

    def repair_portion_size_fact_arity(self, statements: List[str]) -> List[str]:
        """
        Replace a unary portion-size fact when extraction emitted one
        fact but the same subject has an explicit eating fact with an object.

        Example:
          (EatsFrequently abebe pasta) + (EatsLargePortions abebe)
          -> (EatsLargePortions abebe pasta)

        Keeping both arities would create an unstable predicate schema, so the
        repair is applied only when exactly one food object is recoverable.
        """
        eating_objects_by_subject: dict[str, list[str]] = {}
        existing_atoms: set[str] = set()

        repaired_statements: List[str] = []
        for statement in statements:
            repaired = statement
            for head, args in self._simple_fact_atoms(statement):
                atom = f"({head} {' '.join(args)})"
                existing_atoms.add(atom)
                if head in {"Eats", "EatsFrequently"} and len(args) >= 2:
                    eating_objects_by_subject.setdefault(args[0], [])
                    if args[1] not in eating_objects_by_subject[args[0]]:
                        eating_objects_by_subject[args[0]].append(args[1])

        for statement in statements:
            repaired = statement
            for head, args in self._simple_fact_atoms(statement):
                if not self._is_portion_size_head(head) or len(args) != 1:
                    continue
                subject = args[0]
                objects = eating_objects_by_subject.get(subject, [])
                if len(objects) != 1:
                    continue
                repaired_atom = f"({head} {subject} {objects[0]})"
                repaired = re.sub(
                    rf"\({re.escape(head)}\s+{re.escape(subject)}\)",
                    repaired_atom,
                    repaired,
                    count=1,
                )
                existing_atoms.add(repaired_atom)
            repaired_statements.append(repaired)

        return self.dedupe_preserve_order(repaired_statements)

    def _simple_fact_atoms(self, statement: str) -> List[tuple[str, List[str]]]:
        if "Implication" in statement or "(Not " in statement:
            return []
        atoms: List[tuple[str, List[str]]] = []
        for match in re.finditer(
            r"\(([A-Z][A-Za-z0-9_]*)\s+([^()]+?)\)\s+\(STV",
            statement,
        ):
            head = match.group(1)
            if head in self.STRUCTURAL_HEADS:
                continue
            args = [arg for arg in match.group(2).split() if arg]
            if args:
                atoms.append((head, args))
        return atoms

    def _is_portion_size_head(self, head: str) -> bool:
        return bool(
            re.search(r"(Large|Small|Moderate|Huge|Tiny)Portions$", head)
        )

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
        token = token.strip()
        if not token:
            return token
        token = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", token)
        token = token.replace("-", "_")
        token = re.sub(r"[^A-Za-z0-9_]", "_", token)
        token = re.sub(r"_+", "_", token).strip("_")
        token = token.lower()
        if lemmatize and token and not protect:
            token = "_".join(
                self.singularize(part) for part in token.split("_") if part
            )
        return token

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
                heuristic = self.build_heuristic_question_queries(question)
                return self.filter_query_candidates(
                    self.dedupe_preserve_order(semantic_fallback + heuristic)
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
            # Add heuristic query patterns (colleague's approach)
            heuristic = self.build_heuristic_question_queries(question)
            ordered.extend(heuristic)
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

    def build_heuristic_question_queries(self, question: str) -> List[str]:
        """
        Generate PLN queries from question patterns (colleague's approach).
        Patterns like "Does X have Y?" → (: $prf (HasA X Y) $tv)
        """
        normalized = self.normalize_text(question).strip("? ")
        patterns = [
            # (regex pattern, predicate head) - more specific first, generic last
            (r"^(?:is|are|was|were)\s+(.+?)\s+(?:consuming|consume)\s+(.+)$", "Consumes"),
            (r"^(?:does|do|did|has|have|had)\s+(.+?)\s+consume\s+(.+)$", "Consumes"),
            (r"^(?:does|do|did|has|have|had)\s+(.+?)\s+(?:eat|eats)\s+(.+)$", "Eats"),
            (r"^(?:does|do|did|has|have|had)\s+(.+?)\s+(?:have|has|get)\s+(.+)$", "HasA"),
            (r"^(?:is|are|was|were)\s+(.+?)\s+(?:a|an)\s+([a-z][a-z0-9_]*)\s*$", "IsA"),  # "Is X a Y"
            (r"^(?:is|are)\s+(.+?)\s+(?:smart|intelligent|capable|able|mortal)$", "IsA"),
        ]

        queries: List[str] = []
        for pattern, head in patterns:
            match = re.match(pattern, normalized)
            if not match:
                continue
            subject = self.canonical_phrase(match.group(1))
            target = self.canonical_phrase(match.group(2)) if match.lastindex >= 2 else None
            if subject:
                if target:
                    queries.append(f"(: $prf ({head} {subject} {target}) $tv)")
                else:
                    queries.append(f"(: $prf ({head} {subject}) $tv)")
        return queries

    def canonical_phrase(self, phrase: str) -> str:
        """Convert a phrase to canonical underscore-separated form."""
        tokens = [
            token
            for token in self.normalize_text(phrase).split()
            if token not in {"a", "an", "the", "of", "in", "on", "at", "is", "are", "was", "were",
                           "do", "does", "did", "have", "has", "had", "consuming", "consume",
                           "eating", "eat", "eats", "smart", "intelligent", "capable", "able",
                           "mortal"}
        ]
        normalized = [self.canonical_symbol(token, lemmatize=True) for token in tokens if token]
        return "_".join(token for token in normalized if token)

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

    def build_query_hints(
        self,
        original: str,
        normalized: str,
        predicates: List[str],
    ) -> List[str]:
        hints: List[str] = []
        tokens = normalized.split()
        if not tokens:
            return hints

        if tokens[0] in {
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
        }:
            hints.append(
                "; query intent: yes/no question - prefer a direct provable query"
            )
        elif any(marker in tokens for marker in self.QUERY_MARKERS):
            hints.append(
                "; query intent: open question - prefer a variable-bearing query"
            )

        if any(
            token in {"any", "anything", "someone", "somebody", "something"}
            for token in tokens
        ):
            hints.append(
                "; existential wording may justify a helper predicate when a direct query shape is not derivable"
            )

        if "not" in tokens or "never" in tokens:
            hints.append(
                "; use negation only if it is directly supported by explicit facts or rules"
            )

        if predicates:
            hints.append(
                f"; prioritize these predicate heads first: {', '.join(predicates[:5])}"
            )

        if original.endswith("?"):
            hints.append(
                "; preserve the question semantics while keeping the final query executable"
            )

        return hints

    def _materialize_grounded_premise_facts(
        self, text: str, statements: List[str]
    ) -> List[str]:
        """
        Materialize grounded premises AND conclusions from rules.

        Premise materialization:
          "People who eat fish are smart. Kebede eats fish."
          → materializes (EatsFish kebede) as direct fact.

        Conclusion materialization (when all premises are grounded):
          Rule: EatsFrequently($x, pasta) → ConsumesExcessiveCarbs($x)
          Fact: EatsFrequently(abebe, pasta) exists
          → materializes (ConsumesExcessiveCarbs abebe) directly.
        """
        # Retained temporarily for compatibility with external callers. Premise
        # invention is proof-unsafe and this method must never emit atoms.
        return []

        normalized = self.normalize_text(text)
        # Skip materialization for purely definitional rules
        conditional_phrases = (" indicate ", " indicates ", " implies ",
                              " suggest ", " suggests ", " if ", " when ",
                              " should ", " would ", " could ")
        skip_materialization = any(phrase in normalized for phrase in conditional_phrases)

        tokens = set(normalized.split())
        facts: List[str] = []

        # First pass: collect all existing grounded facts from statements
        existing_facts: set[str] = set()
        negated_facts: set[str] = set()  # Atoms that are negated (NOT True)
        for stmt in statements:
            # Skip rules/implications
            if "Implication" in stmt:
                continue
            # Check for negated atoms: (Not (Predicate ...))
            not_match = re.search(r"\(Not\s+\(([A-Za-z][A-Za-z0-9_]*)\s+([^\)]+)\)\)", stmt)
            if not_match:
                pred = not_match.group(1)
                args = not_match.group(2).strip()
                negated_facts.add(f"({pred} {args})")
                continue
            # Match typed facts: (: name (Predicate arg1 arg2) (STV ...))
            m = re.search(r"\(:\s+\S+\s+\(([A-Za-z][A-Za-z0-9_]*)\s+([^\)]+)\)\s+\(", stmt)
            if m:
                pred = m.group(1)
                args = m.group(2).strip()
                existing_facts.add(f"({pred} {args})")
                continue
            # Match bare atoms: (IsA X Y) or (Predicate arg1 arg2)
            bare_match = re.match(r"^\(([A-Za-z][A-Za-z0-9_]*)\s+([^\)]+)\)$", stmt.strip())
            if bare_match:
                pred = bare_match.group(1)
                args = bare_match.group(2).strip()
                existing_facts.add(f"({pred} {args})")

        for stmt in statements:
            # Extract (Implication (Premises ...) (Conclusions ...))
            # Note: Conclusions may have ) from STV following it, so we match conservatively
            m = re.search(r"\(Implication\s+\(Premises\s+(.+?)\)\s+\(Conclusions\s+((?:[^()]|\([^()]*\))+)\)\)", stmt)
            if not m:
                continue

            premises_blob = m.group(1)
            conclusions_blob = m.group(2)

            # Parse premises to extract atoms, variables, and negation
            premise_atoms: List[tuple[str, List[str], bool]] = []  # [(predicate, [args], negated)]
            bindings: dict[str, str] = {}  # variable -> grounded term
            all_grounded = True

            # Parse premises with negation awareness
            # Pattern to match atoms: optionally preceded by (Not ...)
            # Simple atoms: (Predicate arg1 arg2)
            # Negated atoms: (Not (Predicate arg1 arg2))
            pos = 0
            while pos < len(premises_blob):
                # Try to match (Not (Predicate ...))
                not_match = re.match(r"\s*\(Not\s+\(([A-Za-z][A-Za-z0-9_]*)\s+([^\)]+)\)\)", premises_blob[pos:])
                if not_match:
                    head = not_match.group(1)
                    arg_blob = not_match.group(2).strip()
                    args = [a for a in arg_blob.split() if a]
                    premise_atoms.append((head, args, True))  # True = negated
                    pos += not_match.end()
                    continue

                # Try to match (Predicate ...)
                atom_match = re.match(r"\s*\(([A-Za-z][A-Za-z0-9_]*)\s+([^\)]+)\)", premises_blob[pos:])
                if atom_match:
                    head = atom_match.group(1)
                    arg_blob = atom_match.group(2).strip()
                    args = [a for a in arg_blob.split() if a]
                    if args:  # Only add if we have args
                        premise_atoms.append((head, args, False))  # False = not negated
                    pos += atom_match.end()
                    continue

                # Skip this character
                pos += 1

            if not premise_atoms:
                continue

            # Build predicate -> patterns from existing facts
            # e.g., "EatsFrequently abebe pasta" → pattern for EatsFrequently = [abebe, pasta]
            fact_predicate_patterns: dict[str, List[str]] = {}
            for fact in existing_facts:
                m = re.match(r"\(([A-Za-z][A-Za-z0-9_]*)\s+(.+)\)", fact)
                if m:
                    pred = m.group(1)
                    args = [a.strip() for a in m.group(2).split() if a.strip()]
                    if pred not in fact_predicate_patterns:
                        fact_predicate_patterns[pred] = args

            # Bind variables based on patterns and tokens
            all_grounded = True
            for head, args, negated in premise_atoms:
                for a in args:
                    if a.startswith(("$", "?")):
                        var_name = a[1:]  # strip $ or ?
                        if a in bindings:
                            continue  # already bound

                        # Strategy 1: check tokens directly
                        canonical_var = self.canonical_symbol(var_name, lemmatize=True)
                        if canonical_var in tokens:
                            bindings[a] = canonical_var
                            continue

                        # Strategy 2: check existing facts for matching predicate patterns
                        if head in fact_predicate_patterns:
                            patterns = fact_predicate_patterns[head]
                            # Find position of variable in premise (0-indexed)
                            var_pos = args.index(a) if a in args else -1
                            if var_pos >= 0 and var_pos < len(patterns):
                                entity = patterns[var_pos]
                                if entity in tokens or self.canonical_symbol(entity) in tokens:
                                    bindings[a] = entity
                                    continue

                        # Strategy 3: look for any entity in text that could fill this role
                        for token in tokens:
                            skip_words = (
                                self.STOPWORDS
                                | self.GENERIC_SORTALS
                                | self._schema_alignment.generic_terms
                                | {
                                    "almost",
                                    "any",
                                    "each",
                                    "every",
                                    "many",
                                    "much",
                                    "only",
                                }
                            )
                            if token.lower() in skip_words or len(token) < 3:
                                continue
                            # Check if this entity appears in any existing fact with this predicate
                            for fact in existing_facts:
                                if head in fact and token in fact:
                                    bindings[a] = token
                                    break
                            if a in bindings:
                                break

                        if a not in bindings:
                            all_grounded = False
                    else:
                        canon = self.canonical_symbol(a)
                        if a != canon:
                            bindings[a] = canon

            if not all_grounded:
                continue

            # Check if all premise atoms are satisfied
            premises_satisfied = True
            for head, args, negated in premise_atoms:
                bound_args = [bindings.get(a, self.canonical_symbol(a)) for a in args]
                fact_atom = f"({head} {' '.join(bound_args)})"
                if negated:
                    if fact_atom not in negated_facts:
                        premises_satisfied = False
                        break
                else:
                    if fact_atom not in existing_facts:
                        premises_satisfied = False
                        break

            if not premises_satisfied:
                # Materialize premises (colleague's approach)
                if not skip_materialization:
                    for head, args, negated in premise_atoms:
                        if negated:
                            continue  # Don't materialize negated premises as positive facts
                        bound_args = [bindings.get(a, self.canonical_symbol(a)) for a in args]
                        fact_atom = f"({head} {' '.join(bound_args)})"
                        fact_name = f"materialized_{self.canonical_symbol(head)}_fact"
                        facts.append(f"(: {fact_name} {fact_atom} (STV 1.0 1.0))")
                continue

            # All premises satisfied → materialize conclusions
            for conc_match in re.finditer(r"\(([A-Za-z][A-Za-z0-9_]*)\s+([^()]+?)\)", conclusions_blob):
                conc_head = conc_match.group(1)
                conc_args_raw = conc_match.group(2).strip()
                conc_args = [a for a in conc_args_raw.split() if a]

                bound_args = []
                for a in conc_args:
                    if a.startswith(("$", "?")):
                        # Look up binding for this variable
                        if a in bindings:
                            bound_args.append(bindings[a])
                        else:
                            name = self.canonical_symbol(a[1:])
                            if name in tokens:
                                bound_args.append(name)
                            else:
                                bound_args = []
                                break
                    else:
                        bound_args.append(self.canonical_symbol(a))

                if bound_args:
                    conc_atom = f"({conc_head} {' '.join(bound_args)})"
                    if conc_atom not in existing_facts:
                        conc_name = f"materialized_{self.canonical_symbol(conc_head)}_fact"
                        facts.append(f"(: {conc_name} {conc_atom} (STV 1.0 1.0))")

        return self.dedupe_preserve_order(facts)

    def dedupe_preserve_order(self, items: List[str]) -> List[str]:
        seen = set()
        deduped: List[str] = []
        for item in items:
            clean = " ".join(item.split())
            if clean and clean not in seen:
                seen.add(clean)
                deduped.append(clean)
        return deduped
