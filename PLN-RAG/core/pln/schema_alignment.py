from __future__ import annotations

import re
from typing import Iterable, List

from core.pln.symbol_normalization import canonical_symbol


class PLNSchemaAligner:
    """Predicate-signature utilities and conservative schema bridge generation."""

    BRIDGE_STV = "(STV 0.9 0.8)"
    BRIDGE_MAX_PER_CHUNK = 8
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
    # Function words only. This deliberately excludes domain vocabulary: the list
    # previously also held `consume`, `consumption`, `intake`, `risk`, `effect`,
    # `high`, `low`, `increase` and `reduce`, which are nutrition-benchmark words.
    # Classifying those as content-free is what let the bundled cases pass while
    # the same scoring misjudged every other domain — `intake` is generic in a diet
    # paragraph and highly specific in a pharmacokinetics one.
    GENERIC_TERMS = {
        "a",
        "an",
        "and",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
    def __init__(self, structural_heads: Iterable[str]):
        self.structural_heads = set(structural_heads)
        self.skip_heads = self.structural_heads | {"IsA"}

    @property
    def generic_terms(self) -> set[str]:
        return set(self.GENERIC_TERMS)


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



    def bridge_domain_terms(self, terms: set[str]) -> set[str]:
        return {
            term
            for term in terms
            if term not in self.GENERIC_TERMS and not term.isdigit()
        }


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
