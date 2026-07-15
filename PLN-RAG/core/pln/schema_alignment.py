from __future__ import annotations

import re
from typing import Iterable, List

from core.pln.symbol_normalization import canonical_symbol


class PLNSchemaAligner:
    """Predicate-signature utilities and conservative schema bridge generation."""

    BRIDGE_STV = "(STV 0.9 0.8)"
    BRIDGE_MAX_PER_CHUNK = 8
    BRIDGE_MIN_TERM_JACCARD = 0.75
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
        "minimum",
        "maximum",
    }
    GENERIC_TERMS = {
        "at",
        "be",
        "consistently",
        "consume",
        "consumption",
        "effect",
        "frequently",
        "has",
        "have",
        "high",
        "increase",
        "increased",
        "intake",
        "is",
        "lead",
        "leads",
        "low",
        "of",
        "reduce",
        "reduced",
        "risk",
        "to",
    }
    def __init__(self, structural_heads: Iterable[str]):
        self.structural_heads = set(structural_heads)
        self.skip_heads = self.structural_heads | {"IsA"}

    @property
    def generic_terms(self) -> set[str]:
        return set(self.GENERIC_TERMS)

    def build_bridges(
        self,
        statements: List[str],
        context: List[str],
    ) -> tuple[List[str], List[dict]]:
        sources = self.collect_bridge_source_signatures(statements, "current")
        sources.extend(self.collect_bridge_source_signatures(context, "context"))
        targets = self.collect_premise_signatures(statements, origin="current")
        targets.extend(self.collect_premise_signatures(context, origin="context"))
        negated_facts = self.collect_negated_fact_signatures(statements + context)

        bridges: List[str] = []
        decisions: List[dict] = []
        seen: set[str] = set()

        for source in sources:
            for target in targets:
                if source["origin"] == "context" and target["origin"] == "context":
                    continue
                if not self.is_safe_bridge_pair(source, target):
                    continue
                if self.bridge_conflicts_with_negated_fact(target, negated_facts):
                    decisions.append(
                        {
                            "action": "bridge_rejected",
                            "reason": "explicit_negation_conflict",
                            "source": self.signature_atom(source),
                            "target": self.signature_atom(target),
                            "source_origin": source["origin"],
                            "target_origin": target["origin"],
                        }
                    )
                    continue

                statement = self.bridge_statement(source, target)
                if not statement or statement in seen:
                    continue
                seen.add(statement)
                bridges.append(statement)
                decisions.append(
                    {
                        "action": "bridge_added",
                        "source": self.signature_atom(source),
                        "target": self.signature_atom(target),
                        "source_origin": source["origin"],
                        "target_origin": target["origin"],
                        "score": self.bridge_similarity_score(source, target),
                        "statement": statement,
                    }
                )
                if len(bridges) >= self.BRIDGE_MAX_PER_CHUNK:
                    return bridges, decisions

        return bridges, decisions

    def collect_bridge_source_signatures(
        self,
        statements: List[str],
        origin: str,
    ) -> List[dict]:
        signatures: List[dict] = []
        for atom in statements:
            for signature in self.extract_fact_signatures(atom):
                tagged = dict(signature)
                tagged.update({"origin": origin, "role": "fact"})
                signatures.append(tagged)
            for signature in self.extract_conclusion_signatures(atom):
                tagged = dict(signature)
                tagged.update({"origin": origin, "role": "conclusion"})
                signatures.append(tagged)
        return self.dedupe_signatures(signatures)

    def collect_premise_signatures(
        self,
        statements: List[str],
        origin: str,
    ) -> List[dict]:
        signatures: List[dict] = []
        for statement in statements:
            for block in self.extract_named_blocks(statement, "Premises"):
                for atom in self.simple_atoms_from_block(block):
                    parsed = self.parse_simple_atom(atom)
                    if not parsed:
                        continue
                    if parsed["head"] in self.skip_heads:
                        continue
                    tagged = dict(parsed)
                    tagged.update({"origin": origin, "role": "premise"})
                    signatures.append(tagged)
        return self.dedupe_signatures(signatures)

    def is_safe_bridge_pair(self, source: dict, target: dict) -> bool:
        if source["head"] == target["head"]:
            return False
        if source["head"] in self.skip_heads:
            return False
        if target["head"] in self.skip_heads:
            return False
        if source["arity"] != target["arity"]:
            return False
        if source["arity"] == 0:
            return False

        source_terms = self.normalized_head_terms(source["head"])
        target_terms = self.normalized_head_terms(target["head"])
        if (source_terms | target_terms) & self.COMPARATOR_TERMS:
            return False
        overlap = source_terms.intersection(target_terms)
        score = self.bridge_similarity_score(source, target)
        if score < 6 or len(overlap) < 2:
            return False
        if self.bridge_term_jaccard(source_terms, target_terms) < (
            self.BRIDGE_MIN_TERM_JACCARD
        ):
            return False
        source_domain = self.bridge_domain_terms(source_terms)
        target_domain = self.bridge_domain_terms(target_terms)
        return bool(source_domain and target_domain and source_domain & target_domain)

    def bridge_domain_terms(self, terms: set[str]) -> set[str]:
        return {
            term
            for term in terms
            if term not in self.GENERIC_TERMS and not term.isdigit()
        }

    def bridge_term_jaccard(self, source_terms: set[str], target_terms: set[str]) -> float:
        union = source_terms | target_terms
        if not union:
            return 0.0
        return len(source_terms & target_terms) / len(union)

    def bridge_conflicts_with_negated_fact(
        self,
        target: dict,
        negated_facts: List[dict],
    ) -> bool:
        for negated in negated_facts:
            if negated["head"] != target["head"]:
                continue
            if negated["arity"] != target["arity"]:
                continue
            if self.signature_args_may_overlap(target["args"], negated["args"]):
                return True
        return False

    def signature_args_may_overlap(
        self,
        left_args: List[str],
        right_args: List[str],
    ) -> bool:
        for left, right in zip(left_args, right_args):
            if left.startswith(("$", "?")) or right.startswith(("$", "?")):
                continue
            if left != right:
                return False
        return True

    def bridge_similarity_score(self, source: dict, target: dict) -> int:
        source_terms = self.normalized_head_terms(source["head"])
        target_terms = self.normalized_head_terms(target["head"])
        overlap = source_terms.intersection(target_terms)
        if not overlap:
            return 0
        score = len(overlap) * 3
        if source["arity"] == target["arity"]:
            score += 2
        return score

    def normalized_terms(self, terms: Iterable[str]) -> set[str]:
        return {canonical_symbol(term) for term in terms if term}

    def normalized_head_terms(self, head: str) -> set[str]:
        terms: set[str] = set()
        for term in self.split_symbol_terms(head):
            if not term or term in {"has", "have", "is", "at", "of", "to"}:
                continue
            terms.add(term)
        return terms

    def split_symbol_terms(self, symbol: str) -> List[str]:
        symbol = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", symbol)
        parts = re.split(r"[^A-Za-z0-9]+", symbol)
        return [
            canonical_symbol(part, lemmatize=True)
            for part in parts
            if part
        ]

    def bridge_statement(self, source: dict, target: dict) -> str:
        variables = self.bridge_variables(source["arity"])
        source_atom = f"({source['head']} {' '.join(variables)})"
        target_atom = f"({target['head']} {' '.join(variables)})"
        name = self.bridge_name(source["head"], target["head"])
        return (
            f"(: {name} "
            f"(Implication (Premises {source_atom}) "
            f"(Conclusions {target_atom})) {self.BRIDGE_STV})"
        )

    def bridge_variables(self, arity: int) -> List[str]:
        names = ["$x", "$y", "$z", "$a", "$b", "$c"]
        return names[:arity]

    def bridge_name(self, source_head: str, target_head: str) -> str:
        source = canonical_symbol(source_head, lemmatize=False)
        target = canonical_symbol(target_head, lemmatize=False)
        return f"{source}_to_{target}_bridge"[:80]

    def signature_atom(self, signature: dict) -> str:
        args = " ".join(signature["args"])
        return f"({signature['head']} {args})" if args else f"({signature['head']})"

    def dedupe_signatures(self, signatures: List[dict]) -> List[dict]:
        seen: set[tuple] = set()
        result: List[dict] = []
        for signature in signatures:
            key = (
                signature.get("head"),
                tuple(signature.get("args", [])),
                signature.get("origin"),
                signature.get("role"),
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(signature)
        return result

    def extract_named_blocks(self, text: str, name: str) -> List[str]:
        blocks: List[str] = []
        marker = f"({name}"
        start = 0
        while True:
            idx = text.find(marker, start)
            if idx < 0:
                break
            depth = 0
            body_start = idx + len(marker)
            for pos in range(idx, len(text)):
                if text[pos] == "(":
                    depth += 1
                elif text[pos] == ")":
                    depth -= 1
                    if depth == 0:
                        blocks.append(text[body_start:pos].strip())
                        start = pos + 1
                        break
            else:
                break
        return blocks

    def simple_atoms_from_block(self, block: str) -> List[str]:
        atoms: List[str] = []
        for match in re.finditer(
            r"\(([A-Za-z][A-Za-z0-9_]*)((?:\s+[^()\s]+)+)\)",
            block,
        ):
            atom = match.group(0)
            parsed = self.parse_simple_atom(atom)
            if parsed and parsed["head"] not in self.structural_heads:
                atoms.append(atom)
        return atoms

    def collect_available_signatures(
        self,
        statements: List[str],
        context: List[str],
    ) -> tuple[list[dict], list[dict]]:
        facts: list[dict] = []
        conclusions: list[dict] = []
        for atom in statements + context:
            facts.extend(self.extract_fact_signatures(atom))
            conclusions.extend(self.extract_conclusion_signatures(atom))
        return facts, conclusions

    def extract_fact_signatures(self, text: str) -> list[dict]:
        signatures: list[dict] = []
        for match in re.finditer(
            r"\(:\s+[^\s()]+\s+(\([^()]+\))\s+\((?:STV|PointMass|ParticleFromNormal|ParticleFromPairs)",
            text,
        ):
            parsed = self.parse_simple_atom(match.group(1))
            if parsed and parsed["head"] != "Implication":
                signatures.append(parsed)
        return signatures

    def collect_negated_fact_signatures(self, items: List[str]) -> list[dict]:
        signatures: list[dict] = []
        for item in items:
            signatures.extend(self.extract_negated_fact_signatures(item))
        return signatures

    def extract_negated_fact_signatures(self, text: str) -> list[dict]:
        signatures: list[dict] = []
        for match in re.finditer(
            r"\(:\s+[^\s()]+\s+\(Not\s+(\([^()]+\))\)\s+\((?:STV|PointMass|ParticleFromNormal|ParticleFromPairs)",
            text,
        ):
            parsed = self.parse_simple_atom(match.group(1))
            if parsed:
                signatures.append(parsed)
        return signatures

    def extract_conclusion_signatures(self, text: str) -> list[dict]:
        signatures: list[dict] = []
        for block in re.finditer(r"\(Conclusions\s+((?:\([^()]+\)\s*)+)\)", text):
            for atom in re.finditer(r"\([^()]+\)", block.group(1)):
                parsed = self.parse_simple_atom(atom.group(0))
                if parsed:
                    signatures.append(parsed)
        return signatures

    def parse_query_signature(self, query: str) -> dict | None:
        match = re.search(
            r"\(:\s+[^\s()]+\s+(\([^()]+\))\s+\$?[A-Za-z_][A-Za-z0-9_]*\)",
            query,
        )
        if not match:
            return None
        return self.parse_simple_atom(match.group(1))

    def parse_simple_atom(self, atom: str) -> dict | None:
        match = re.fullmatch(
            r"\(([A-Za-z][A-Za-z0-9_]*)((?:\s+[^()\s]+)*)\)",
            atom.strip(),
        )
        if not match:
            return None
        head = match.group(1)
        args = [part for part in match.group(2).split() if part]
        return {
            "head": head,
            "args": args,
            "arity": len(args),
            "variables": [arg for arg in args if arg.startswith(("$", "?"))],
        }

    def signature_to_query(self, signature: dict) -> str:
        args = " ".join(signature["args"])
        return f"(: $prf ({signature['head']} {args}) $tv)"
