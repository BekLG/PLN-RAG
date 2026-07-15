import os
import json
import re
import threading
from dataclasses import dataclass, field
from typing import Any, List
from config import get_settings
from core.pln.symbol_normalization import canonical_symbol

from pettachainer.pettachainer import PeTTaChainer


@dataclass
class ProofOutcome:
    status: str = "unknown"
    positive_query: str = ""
    negative_query: str = ""
    positive_proof: List[str] = field(default_factory=list)
    negative_proof: List[str] = field(default_factory=list)
    proof_validated: bool = True
    support_kind: str = "unknown"

    @property
    def proof(self) -> List[str]:
        return self.positive_proof + self.negative_proof


class Reasoner:
    """
    Owns all atomspace operations. Nothing else in the system
    calls add_atom or query directly.

    Responsibilities:
    - Load atomspace from disk on startup
    - Add statements coming from the parser
    - Execute queries and return proof traces
    - Persist new atoms to disk
    """

    def __init__(self):
        cfg = get_settings()
        self._atomspace_path = cfg.atomspace_path
        self._provenance_path = f"{cfg.atomspace_path}.provenance.jsonl"
        self._query_timeout = cfg.chaining_timeout
        self._max_steps = cfg.chaining_max_steps
        self._strict_proof_validation = cfg.strict_proof_validation_enabled
        self._lock = threading.Lock()
        self._handler = PeTTaChainer()
        self._background_files: set[str] = set()
        self._atom_keys: set[str] = set()
        self._load_from_disk()

    def _load_from_disk(self):
        if not os.path.exists(self._atomspace_path):
            os.makedirs(os.path.dirname(self._atomspace_path), exist_ok=True)
            return
        print(f"[Reasoner] Loading atomspace from {self._atomspace_path}...")
        with open(self._atomspace_path, "r", encoding="utf-8") as f:
            for line in f:
                atom = line.strip()
                if atom:
                    try:
                        self._handler.add_atom(atom)
                        self._atom_keys.add(atom)
                    except Exception as e:
                        print(f"[Reasoner] Warning: skipping atom '{atom}': {e}")
        print("[Reasoner] Atomspace loaded.")

    def add_statements(
        self,
        statements: List[str],
        provenance: dict[str, dict[str, Any]] | None = None,
    ) -> List[str]:
        """
        Add parsed MeTTa statements to the atomspace and persist them.
        Returns the list of successfully added atoms.
        """
        added = []
        provenance = provenance or {}
        with self._lock:
            existing = getattr(self, "_atom_keys", None)
            if existing is None:
                existing = set()
                self._atom_keys = existing
            if not existing and os.path.exists(self._atomspace_path):
                with open(self._atomspace_path, "r", encoding="utf-8") as current:
                    existing.update(line.strip() for line in current if line.strip())
            with open(self._atomspace_path, "a", encoding="utf-8") as f, open(
                self._provenance_path, "a", encoding="utf-8"
            ) as provenance_file:
                for stmt in statements:
                    clean = " ".join(stmt.split())
                    if clean in existing:
                        continue
                    try:
                        self._handler.add_atom(clean)
                        f.write(clean + "\n")
                        added.append(clean)
                        existing.add(clean)
                        source = provenance.get(stmt) or provenance.get(clean)
                        if source:
                            provenance_file.write(
                                json.dumps(
                                    {"atom": clean, "source": source},
                                    ensure_ascii=True,
                                )
                                + "\n"
                            )
                    except Exception as e:
                        print(f"[Reasoner] Failed to add atom '{clean}': {e}")
        return added

    def query(self, pln_query: str, seed_terms: List[str] | None = None) -> List[str]:
        """
        Run a PLN query and return proof traces.
        Try exact fact lookup first, then materialize Qdrant-retrieved seed
        facts with PeTTa forward chaining before running the normal query.
        """
        exact = self._query_exact_fact(pln_query)
        if exact:
            return exact

        # Our strict kernel understands explicit negation. PeTTa may interpret a
        # negated premise probabilistically, so any rule path containing Not is
        # proved here and is never delegated as proof authority.
        strict = self._backward_chain(pln_query)
        if strict:
            return strict
        if self._matching_rule_uses_explicit_negation(pln_query):
            return []
        if getattr(self, "_strict_proof_validation", True):
            return []

        try:
            with self._lock:
                forward_ran = self._forward_chain_from_seed_terms(seed_terms or [])
                result = self._handler.query(
                    pln_query,
                    steps=self._max_steps,
                    timeout_sec=0 if forward_ran else self._query_timeout,
                )
            if result:
                return result
        except Exception as e:
            print(f"[Reasoner] Query failed for '{pln_query}': {e}")
        return []

    def query_polarity(
        self,
        pln_query: str,
        seed_terms: List[str] | None = None,
    ) -> ProofOutcome:
        """Check a grounded proposition and its explicit negation."""
        negative_query = self.negated_query(pln_query)
        positive = self.query(pln_query, seed_terms=seed_terms)
        negative = self._explicit_negation_proof(
            negative_query,
            seed_terms=seed_terms,
        )
        if positive and negative:
            status = "both"
        elif positive:
            status = "positive"
        elif negative:
            status = "negative"
        else:
            status = "unknown"
        support_kind = self._support_kind(status, positive, negative)
        return ProofOutcome(
            status=status,
            positive_query=pln_query,
            negative_query=negative_query,
            positive_proof=positive,
            negative_proof=negative,
            proof_validated=True,
            support_kind=support_kind,
        )

    def _support_kind(
        self,
        status: str,
        positive: List[str],
        negative: List[str],
    ) -> str:
        if status == "both":
            return "conflict"
        if status == "negative":
            return "explicit_negative"
        if status != "positive":
            return "unknown"
        for trace in positive:
            for match in re.finditer(r"\(STV\s+([0-9]*\.?[0-9]+)\s+", str(trace)):
                if float(match.group(1)) < 1.0:
                    return "probabilistic"
        return "entailed"

    def grounded_query_atom(self, pln_query: str) -> str:
        return self._extract_grounded_query_atom(pln_query)

    def _explicit_negation_proof(
        self,
        negative_query: str,
        seed_terms: List[str] | None = None,
    ) -> List[str]:
        if not negative_query:
            return []
        if hasattr(self, "_atomspace_path"):
            return self._query_exact_fact(negative_query)
        return self.query(negative_query, seed_terms=seed_terms)

    def negated_query(self, pln_query: str) -> str:
        target = self._extract_grounded_query_atom(pln_query)
        if not target or target.startswith("(Not "):
            return ""
        return f"(: $prf (Not {target}) $tv)"

    def explain_requirements(
        self,
        terms: set[str],
        entity: str = "",
        direction: str = "",
    ) -> List[dict[str, Any]]:
        """Return source-rule requirements relevant to an explanatory question."""
        explanations: List[dict[str, Any]] = []
        terms = {canonical_symbol(term) for term in terms if canonical_symbol(term)}
        generic_terms = {
            "factor", "reason", "increase", "decrease", "reduce", "lower",
            "cause", "lead", "risk", "sufficient", "alone",
        }
        domain_terms = set(terms) - generic_terms
        for rule in self._extract_rules(self.get_atoms()):
            for conclusion in rule["conclusions"]:
                if not isinstance(conclusion, list) or not conclusion:
                    continue
                head_terms = self._head_terms(str(conclusion[0]))
                if not head_terms.intersection(terms):
                    continue
                head_domain_terms = head_terms - generic_terms
                if domain_terms and not head_domain_terms.intersection(domain_terms):
                    continue
                if direction == "reduce" and not head_terms.intersection(
                    {"reduce", "decrease", "lower", "prevent", "protect", "mitigate"}
                ):
                    continue
                bindings: dict[str, str] = {}
                if entity:
                    for arg in conclusion[1:]:
                        if self._is_variable(arg):
                            bindings[str(arg)] = entity
                grounded_target = self._apply_bindings_expr(conclusion, bindings)
                requirements = [
                    self._requirement_record(premise, bindings)
                    for premise in rule["premises"]
                ]
                explanations.append(
                    {
                        "target": self._serialize_expr(grounded_target),
                        "rule": rule["raw"],
                        "requirements": requirements,
                    }
                )
        return explanations

    def _requirement_record(self, premise: Any, bindings: dict) -> dict[str, Any]:
        grounded = self._apply_bindings_expr(premise, bindings)
        if isinstance(grounded, list) and grounded:
            head = str(grounded[0])
            if head in {"And", "and", "Premises"}:
                children = [self._requirement_record(item, bindings) for item in grounded[1:]]
                return {
                    "kind": "and",
                    "status": "positive" if all(item["status"] == "positive" for item in children) else "unknown",
                    "children": children,
                }
            if head in {"Or", "or"}:
                options = [self._requirement_record(item, bindings) for item in grounded[1:]]
                statuses = {item["status"] for item in options}
                status = "positive" if "positive" in statuses else (
                    "negative" if statuses == {"negative"} else "unknown"
                )
                return {"kind": "or", "status": status, "options": options}

        atom = self._serialize_expr(grounded)
        query = f"(: $prf {atom} $tv)"
        if atom.startswith("(Not "):
            proof = self.query(query)
            status = "positive" if proof else "unknown"
        else:
            positive = self.query(query)
            negative_query = self.negated_query(query)
            if negative_query and hasattr(self, "_atomspace_path"):
                negative = self._query_exact_fact(negative_query)
            elif negative_query:
                negative = self.query(negative_query)
            else:
                negative = []
            if positive and negative:
                status = "both"
            elif positive:
                status = "positive"
            elif negative:
                status = "negative"
            else:
                status = "unknown"
        return {"kind": "literal", "atom": atom, "status": status}

    def _head_terms(self, head: str) -> set[str]:
        spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", head)
        terms = {
            canonical_symbol(part)
            for part in re.split(r"[^A-Za-z0-9]+", spaced)
            if part
        }
        if "increas" in terms:
            terms.add("increase")
        return terms

    def _forward_chain_from_seed_terms(self, seed_terms: List[str]) -> bool:
        available = self._available_seed_terms(seed_terms)
        if not available:
            return False
        try:
            result = self._handler.forward_chain_from_facts(
                available,
                steps=self._max_steps,
            )
            if result == ["false"]:
                print(
                    "[Reasoner] Forward chaining skipped: no selected seed facts "
                    f"for {available}"
                )
                return False
            return True
        except Exception as exc:
            print(f"[Reasoner] Forward chaining from Qdrant seeds failed: {exc}")
        return False

    def _available_seed_terms(self, seed_terms: List[str]) -> List[str]:
        available: List[str] = []
        seen: set[str] = set()
        for seed in seed_terms:
            clean = " ".join(str(seed).split())
            if not clean or clean in seen or "$" in clean or "?" in clean:
                continue
            if self._query_exact_fact(f"(: $prf {clean} $tv)"):
                seen.add(clean)
                available.append(clean)
        return available

    def _query_exact_fact(self, pln_query: str) -> List[str]:
        target = self._extract_grounded_query_atom(pln_query)
        if not target:
            return []

        for path in self._fact_sources():
            match = self._find_exact_atom_in_file(path, target)
            if match:
                return [match]
        for path in self._fact_sources():
            match = self._find_canonical_atom_in_file(path, target)
            if match:
                return [match]
        return []

    def _extract_grounded_query_atom(self, pln_query: str) -> str:
        match = re.fullmatch(
            r"\(:\s+[$?][^\s]+\s+(\(.+\))\s+[$?][^\s]+\)",
            pln_query.strip(),
        )
        if not match:
            return ""
        atom = match.group(1)
        if "$" in atom or "?" in atom:
            return ""
        return " ".join(atom.split())

    def _fact_sources(self) -> List[str]:
        paths = [self._atomspace_path, *sorted(self._background_files)]
        return [path for path in paths if os.path.exists(path)]

    def _find_exact_atom_in_file(self, path: str, target: str) -> str:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                atom = line.strip()
                if not atom:
                    continue
                body = self._extract_statement_body(atom)
                if body == target:
                    return atom
        return ""

    def _find_canonical_atom_in_file(self, path: str, target: str) -> str:
        target_signature = self._parse_simple_atom(target)
        if not target_signature:
            return ""
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                atom = line.strip()
                if not atom:
                    continue
                body = self._extract_statement_body(atom)
                signature = self._parse_simple_atom(body)
                if not signature:
                    continue
                if self._canonically_same_signature(target_signature, signature):
                    return atom
        return ""

    def _extract_statement_body(self, statement: str) -> str:
        match = re.fullmatch(
            r"\(:\s+[^\s]+\s+(\(.+\))\s+(?:\(STV\s+[^\s]+\s+[^\s]+\)|[$?][^\s]+)\)",
            statement.strip(),
        )
        if not match:
            return ""
        return " ".join(match.group(1).split())

    def _parse_simple_atom(self, atom: str) -> dict | None:
        match = re.fullmatch(r"\(([A-Za-z][A-Za-z0-9_]*)((?:\s+[^()\s]+)*)\)", atom.strip())
        if not match:
            return None
        head = match.group(1)
        args = [part for part in match.group(2).split() if part]
        return {"head": head, "args": args, "arity": len(args)}

    def _canonically_same_signature(self, left: dict, right: dict) -> bool:
        if left["head"] != right["head"] or left["arity"] != right["arity"]:
            return False
        for l_arg, r_arg in zip(left["args"], right["args"]):
            if canonical_symbol(l_arg) != canonical_symbol(r_arg):
                return False
        return True

    def reset(self):
        """Clear the in-memory atomspace and wipe the persistence file."""
        with self._lock:
            self._handler = PeTTaChainer()
            self._background_files = set()
            self._atom_keys = set()
            if os.path.exists(self._atomspace_path):
                os.remove(self._atomspace_path)
            if os.path.exists(self._provenance_path):
                os.remove(self._provenance_path)
        print("[Reasoner] Atomspace reset.")

    @property
    def size(self) -> int:
        """Approximate atom count (line count of persistence file)."""
        if not os.path.exists(self._atomspace_path):
            return 0
        with open(self._atomspace_path) as f:
            return sum(1 for line in f if line.strip())

    def get_atoms(self, pattern: str | None = None) -> List[str]:
        """Return all atoms, optionally filtered by pattern substring."""
        if not os.path.exists(self._atomspace_path):
            return []
        with open(self._atomspace_path, "r", encoding="utf-8") as f:
            atoms = [line.strip() for line in f if line.strip()]
        if pattern:
            return [atom for atom in atoms if pattern in atom]
        return atoms

    def predicate_arities(self) -> dict[str, set[int]]:
        """Return observed predicate arities from facts and rule atoms."""
        arities: dict[str, set[int]] = {}

        def record(expr: Any) -> None:
            if not isinstance(expr, list) or not expr:
                return
            head = expr[0]
            if head in {"Not"} and len(expr) == 2:
                record(expr[1])
                return
            if isinstance(head, str) and head not in {
                ":",
                "Implication",
                "Premises",
                "Conclusions",
                "And",
                "Or",
                "STV",
            }:
                arities.setdefault(head, set()).add(len(expr) - 1)

        for fact_expr, _raw in self._fact_exprs():
            record(fact_expr)
        for rule in self._extract_rules(self.get_atoms()):
            for premise in rule["premises"]:
                record(premise)
            for conclusion in rule["conclusions"]:
                record(conclusion)
        return arities

    def proposition_signatures(self) -> List[dict[str, Any]]:
        """Return trusted fact and rule-literal signatures for query compilation."""
        signatures: List[dict[str, Any]] = []
        seen: set[tuple[str, tuple[str, ...], bool, str]] = set()

        def record(expr: Any, role: str, negated: bool = False) -> None:
            if not isinstance(expr, list) or not expr:
                return
            head = expr[0]
            if head in {"Not", "not"} and len(expr) == 2:
                record(expr[1], role, True)
                return
            if head in {"And", "and", "Or", "or", "Premises", "Conclusions"}:
                for child in expr[1:]:
                    record(child, role, negated)
                return
            if not isinstance(head, str) or head in {
                ":", "Implication", "STV", "PointMass", "ParticleFromNormal",
                "ParticleFromPairs",
            }:
                return
            args = tuple(str(arg) for arg in expr[1:] if isinstance(arg, str))
            if len(args) != len(expr) - 1:
                return
            key = (head, args, negated, role)
            if key in seen:
                return
            seen.add(key)
            signatures.append(
                {
                    "head": head,
                    "args": list(args),
                    "arity": len(args),
                    "variables": [arg for arg in args if self._is_variable(arg)],
                    "negated": negated,
                    "role": role,
                }
            )

        for fact_expr, _raw in self._fact_exprs():
            record(fact_expr, "fact")
        for rule in self._extract_rules(self.get_atoms()):
            for premise in rule["premises"]:
                record(premise, "premise")
            for conclusion in rule["conclusions"]:
                record(conclusion, "conclusion")
        return signatures

    def sources_for_proof(self, proof_traces: List[str]) -> List[str]:
        """Resolve proof atoms to exact extraction spans recorded at ingestion."""
        if not os.path.exists(self._provenance_path):
            return []
        proof_atoms = {" ".join(str(item).split()) for item in proof_traces}
        sources: List[str] = []
        seen: set[str] = set()
        with open(self._provenance_path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except (TypeError, ValueError):
                    continue
                if record.get("atom") not in proof_atoms:
                    continue
                source = record.get("source") or {}
                text = str(source.get("text", "")).strip()
                if text and text not in seen:
                    seen.add(text)
                    sources.append(text)
        return sources

    def _parse_sexpr(self, text: str) -> List[Any]:
        """Parse S-expression into nested list."""
        tokens = re.findall(r"\(|\)|[^\s()]+", text)
        roots: List[Any] = []
        stack: List[List[Any]] = []

        for token in tokens:
            if token == "(":
                node: List[Any] = []
                if stack:
                    stack[-1].append(node)
                else:
                    roots.append(node)
                stack.append(node)
                continue
            if token == ")":
                if not stack:
                    return []
                stack.pop()
                continue
            if stack:
                stack[-1].append(token)
            else:
                roots.append(token)

        return roots if not stack else []

    def _extract_rules(self, atoms: List[str]) -> List[dict]:
        """Extract (Implication (Premises ...) (Conclusions ...)) as dicts."""
        rules = []
        seen = set()
        for atom in atoms:
            if "(Implication" not in atom:
                continue
            parsed = self._parse_sexpr(atom)
            for expr in self._walk_sexprs(parsed):
                if isinstance(expr, list) and len(expr) >= 3 and expr[0] == ":":
                    expr = expr[2]
                if not (isinstance(expr, list) and expr and expr[0] == "Implication"):
                    continue
                rule = {"raw": atom, "premises": [], "conclusions": []}
                for block in expr[1:]:
                    if isinstance(block, list) and block:
                        if block[0] == "Premises":
                            rule["premises"] = list(block[1:])
                        elif block[0] == "Conclusions":
                            rule["conclusions"] = list(block[1:])
                if rule["conclusions"]:
                    key = str(rule["premises"]) + str(rule["conclusions"])
                    if key in seen:
                        continue
                    seen.add(key)
                    rules.append(rule)
        return rules

    def _walk_sexprs(self, expr: Any) -> List[Any]:
        found: List[Any] = []
        if isinstance(expr, list):
            found.append(expr)
            for item in expr:
                found.extend(self._walk_sexprs(item))
        return found

    def _match_conclusion(self, query: str, conclusion: Any) -> dict | None:
        """Check if query matches a rule conclusion. Return bindings dict or None."""
        # Parse query: (: $prf (Predicate arg1 arg2 ...) $tv)
        query_match = re.search(r"\(:\s*\$prf\s*(\(.*?\))\s*\$tv\)", query)
        if not query_match:
            return None
        query_expr = self._parse_sexpr(query_match.group(1))
        if len(query_expr) == 1 and isinstance(query_expr[0], list):
            query_expr = query_expr[0]
        if not query_expr or not isinstance(query_expr, list):
            return None

        # conclusion might be a list like (Predicate $var arg) or (Predicate arg)
        conclusion_expr = conclusion if isinstance(conclusion, list) else [conclusion]

        # Compare predicates
        if query_expr[0] != conclusion_expr[0]:
            return None
        if len(query_expr) != len(conclusion_expr):
            return None

        # Collect bindings: match variables in conclusion against query args
        bindings = {}
        for i, (q_arg, c_arg) in enumerate(zip(query_expr[1:], conclusion_expr[1:])):
            if isinstance(c_arg, list):
                # Nested expression in conclusion - can't bind directly
                return None
            if c_arg.startswith(("$", "?")) and c_arg not in {"$prf", "$tv", "?prf", "?tv"}:
                # Variable in conclusion - bind to query arg
                if not q_arg.startswith(("$", "?")):
                    bindings[c_arg] = q_arg
                elif q_arg != c_arg:
                    return None  # Both variables but different names
            elif q_arg != c_arg:
                return None  # Conflict
        return bindings if bindings else {}

    def _apply_bindings(self, expr: Any, bindings: dict) -> str:
        """Apply variable bindings to an expression, return query-style string."""
        if isinstance(expr, list):
            parts = [self._apply_bindings(item, bindings) for item in expr]
            return "(" + " ".join(parts) + ")"
        if isinstance(expr, str):
            return bindings.get(expr, expr)
        return str(expr)

    def _backward_chain(self, query: str, seen: set | None = None) -> List[str]:
        """
        Recursively prove query using backward chaining through Implication rules.
        Returns proof traces for derived conclusions.
        """
        if seen is None:
            seen = set()

        # Normalize query for cycle detection
        normalized = query.strip()
        if normalized in seen:
            return []
        seen.add(normalized)

        proofs = []

        # Find rules where conclusion matches query
        atoms = self.get_atoms()
        rules = self._extract_rules(atoms)

        for rule in rules:
            for conclusion in rule["conclusions"]:
                bindings = self._match_conclusion(query, conclusion)
                if bindings is None:
                    continue

                proof_state = self._prove_premises(rule["premises"], bindings, seen)
                if proof_state is not None:
                    final_bindings, premise_proofs = proof_state
                    # Derive conclusion
                    grounded_conclusion = self._apply_bindings(conclusion, final_bindings)
                    conclusion_atom = self._derived_statement_name(grounded_conclusion)
                    proofs.extend(premise_proofs)
                    proofs.append(rule["raw"])
                    proofs.append(conclusion_atom)

        return self._dedupe(proofs)

    def _prove_premises(
        self,
        premises: List[Any],
        bindings: dict,
        seen: set,
    ) -> tuple[dict, List[str]] | None:
        states: List[tuple[dict, List[str]]] = [(dict(bindings), [])]
        for premise in premises:
            next_states: List[tuple[dict, List[str]]] = []
            for state_bindings, state_proofs in states:
                for new_bindings, proofs in self._prove_premise(
                    premise,
                    state_bindings,
                    seen,
                ):
                    next_states.append((new_bindings, state_proofs + proofs))
            if not next_states:
                return None
            states = next_states[:20]
        return states[0]

    def _prove_premise(
        self,
        premise: Any,
        bindings: dict,
        seen: set,
    ) -> List[tuple[dict, List[str]]]:
        if isinstance(premise, list) and premise:
            head = premise[0]
            if head in {"Or", "or"}:
                alternatives: List[tuple[dict, List[str]]] = []
                for option in premise[1:]:
                    alternatives.extend(
                        self._prove_premise(option, dict(bindings), set(seen))
                    )
                return alternatives[:20]
            if head in {"And", "and"}:
                states: List[tuple[dict, List[str]]] = [(dict(bindings), [])]
                for child in premise[1:]:
                    next_states: List[tuple[dict, List[str]]] = []
                    for state_bindings, state_proofs in states:
                        for updated, proofs in self._prove_premise(
                            child,
                            state_bindings,
                            seen,
                        ):
                            next_states.append((updated, state_proofs + proofs))
                    if not next_states:
                        return []
                    states = next_states[:20]
                return states

        partially_grounded = self._apply_bindings_expr(premise, bindings)
        if not self._variables(partially_grounded):
            proof = self._prove_grounded_premise(partially_grounded, seen)
            return [(dict(bindings), proof)] if proof else []

        matches: List[tuple[dict, List[str]]] = []
        for fact_expr, fact_raw in self._fact_exprs():
            updated = self._unify(partially_grounded, fact_expr, dict(bindings))
            if updated is not None:
                matches.append((updated, [fact_raw]))
        return matches

    def _prove_grounded_premise(self, premise: Any, seen: set) -> List[str]:
        premise_atom = self._serialize_expr(premise)
        if self._is_reflexive_isa(premise):
            return [self._derived_statement_name(premise_atom)]

        premise_query = f"(: $prf {premise_atom} $tv)"
        direct = self._query_exact_fact(premise_query)
        if direct:
            return direct

        # Explicit negation is open-world: only an explicit negative atom may
        # satisfy a negative literal. Failure to prove P never proves Not(P).
        if self._is_explicit_negation(premise):
            return []

        nested = self._backward_chain(premise_query, seen)
        if nested:
            return nested

        if self._matching_rule_uses_explicit_negation(premise_query):
            return []

        try:
            direct_chain = self._handler.query(
                premise_query,
                steps=self._max_steps,
                timeout_sec=self._query_timeout,
            ) or []
        except Exception:
            direct_chain = []
        return direct_chain

    def _is_explicit_negation(self, expr: Any) -> bool:
        return (
            isinstance(expr, list)
            and len(expr) == 2
            and expr[0] in {"Not", "not"}
        )

    def _matching_rule_uses_explicit_negation(self, query: str) -> bool:
        for rule in self._extract_rules(self.get_atoms()):
            if not any(
                self._match_conclusion(query, conclusion) is not None
                for conclusion in rule["conclusions"]
            ):
                continue
            if any(self._contains_explicit_negation(item) for item in rule["premises"]):
                return True
        return False

    def _contains_explicit_negation(self, expr: Any) -> bool:
        if not isinstance(expr, list) or not expr:
            return False
        if expr[0] in {"Not", "not"}:
            return True
        return any(self._contains_explicit_negation(item) for item in expr[1:])

    def _fact_exprs(self) -> List[tuple[Any, str]]:
        facts: List[tuple[Any, str]] = []
        for atom in self.get_atoms():
            body = self._extract_statement_body(atom)
            if not body or "(Implication" in body:
                continue
            parsed = self._parse_sexpr(body)
            if len(parsed) == 1 and isinstance(parsed[0], list):
                parsed = parsed[0]
            if isinstance(parsed, list) and parsed:
                facts.append((parsed, atom))
        return facts

    def _apply_bindings_expr(self, expr: Any, bindings: dict) -> Any:
        if isinstance(expr, list):
            return [self._apply_bindings_expr(item, bindings) for item in expr]
        if isinstance(expr, str):
            return bindings.get(expr, expr)
        return expr

    def _unify(self, pattern: Any, fact: Any, bindings: dict) -> dict | None:
        if isinstance(pattern, str):
            if self._is_variable(pattern):
                bound = bindings.get(pattern)
                if bound is None:
                    bindings[pattern] = fact
                    return bindings
                return bindings if bound == fact else None
            return bindings if pattern == fact else None

        if isinstance(fact, str):
            return None
        if not isinstance(pattern, list) or not isinstance(fact, list):
            return None
        if len(pattern) != len(fact):
            return None

        current = bindings
        for left, right in zip(pattern, fact):
            current = self._unify(left, right, current)
            if current is None:
                return None
        return current

    def _variables(self, expr: Any) -> set[str]:
        if isinstance(expr, str):
            return {expr} if self._is_variable(expr) else set()
        if isinstance(expr, list):
            variables: set[str] = set()
            for item in expr:
                variables.update(self._variables(item))
            return variables
        return set()

    def _is_variable(self, value: Any) -> bool:
        return isinstance(value, str) and value.startswith(("$", "?"))

    def _serialize_expr(self, expr: Any) -> str:
        if isinstance(expr, list):
            return "(" + " ".join(self._serialize_expr(item) for item in expr) + ")"
        return str(expr)

    def _is_reflexive_isa(self, expr: Any) -> bool:
        return (
            isinstance(expr, list)
            and len(expr) == 3
            and expr[0] == "IsA"
            and expr[1] == expr[2]
        )

    def _dedupe(self, items: List[str]) -> List[str]:
        seen = set()
        result = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result

    def _derived_statement_name(self, grounded_conclusion: str) -> str:
        parsed = self._parse_simple_atom(grounded_conclusion)
        if not parsed:
            return f"(: derived {grounded_conclusion} (STV 1.0 1.0))"
        name = "_".join(
            [*parsed["args"][:2], parsed["head"], "derived"]
        ).lower()
        name = re.sub(r"[^a-z0-9_]+", "_", name).strip("_")[:80] or "derived"
        return f"(: {name} {grounded_conclusion} (STV 1.0 1.0))"
