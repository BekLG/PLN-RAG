import re
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from core.symbol_normalization import canonical_symbol, equivalent_symbol


Expression = Any
Bindings = Dict[str, Expression]


def parse_expression(text: str) -> Optional[Expression]:
    tokens = re.findall(r"\(|\)|[^\s()]+", text)
    if not tokens:
        return None

    def parse_at(index: int) -> Tuple[Expression, int]:
        if index >= len(tokens):
            raise ValueError("Unexpected end of expression")
        token = tokens[index]
        if token != "(":
            return token, index + 1

        result: List[Expression] = []
        index += 1
        while index < len(tokens) and tokens[index] != ")":
            value, index = parse_at(index)
            result.append(value)
        if index >= len(tokens):
            raise ValueError("Unclosed expression")
        return result, index + 1

    try:
        expression, next_index = parse_at(0)
    except ValueError:
        return None
    return expression if next_index == len(tokens) else None


class HornFallback:
    def __init__(
        self,
        statements: List[str],
        max_steps: int = 100,
        additional_equivalences: Iterable[Tuple[str, str]] | None = None,
    ):
        self.max_steps = max(1, max_steps)
        self._additional_equivalences = {
            frozenset((canonical_symbol(left), canonical_symbol(right)))
            for left, right in (additional_equivalences or [])
        }
        self.facts: List[Tuple[Expression, str]] = []
        self.rules: List[Tuple[List[Expression], List[Expression], str]] = []
        self._load(statements)

    def prove(self, query_atom: str) -> List[str]:
        goal = parse_expression(query_atom)
        if not isinstance(goal, list) or not goal:
            return []
        result = self._prove(goal, {}, 0, set())
        if not result:
            return []
        _, proof = result
        return self._dedupe(proof)

    def _load(self, statements: List[str]) -> None:
        for raw in statements:
            expression = parse_expression(raw)
            if (
                not isinstance(expression, list)
                or len(expression) < 4
                or expression[0] != ":"
            ):
                continue
            body = expression[2]
            if not isinstance(body, list) or not body:
                continue
            if body[0] != "Implication":
                self.facts.append((body, raw))
                continue
            premises = self._section(body, "Premises")
            conclusions = self._section(body, "Conclusions")
            if conclusions:
                self.rules.append((premises, conclusions, raw))

    def _section(self, implication: List[Expression], name: str) -> List[Expression]:
        for item in implication[1:]:
            if isinstance(item, list) and item and item[0] == name:
                return self._flatten_conjunctions(item[1:])
        return []

    def _flatten_conjunctions(self, expressions: List[Expression]) -> List[Expression]:
        flattened: List[Expression] = []
        for expression in expressions:
            if isinstance(expression, list) and expression and expression[0] == "And":
                flattened.extend(self._flatten_conjunctions(expression[1:]))
            else:
                flattened.append(expression)
        return flattened

    def _prove(
        self,
        goal: Expression,
        bindings: Bindings,
        steps: int,
        trail: set[str],
    ) -> Optional[Tuple[Bindings, List[str]]]:
        return next(
            self._prove_candidates(goal, bindings, steps, trail),
            None,
        )

    def _prove_candidates(
        self,
        goal: Expression,
        bindings: Bindings,
        steps: int,
        trail: set[str],
    ) -> Iterator[Tuple[Bindings, List[str]]]:
        if steps >= self.max_steps:
            return

        grounded_goal = self._substitute(goal, bindings)
        key = self._render(grounded_goal)
        if key in trail:
            return
        next_trail = set(trail)
        next_trail.add(key)

        for fact, raw in self.facts:
            matched = self._unify(grounded_goal, fact, dict(bindings))
            if matched is not None:
                yield matched, [raw]

        for index, (premises, conclusions, raw) in enumerate(self.rules):
            suffix = f"__{steps}_{index}"
            fresh_premises = [self._freshen(item, suffix) for item in premises]
            for conclusion in conclusions:
                fresh_conclusion = self._freshen(conclusion, suffix)
                matched = self._unify(
                    fresh_conclusion,
                    grounded_goal,
                    dict(bindings),
                )
                if matched is None:
                    continue
                for result_bindings, proof in self._prove_all_candidates(
                    fresh_premises,
                    matched,
                    steps + 1,
                    next_trail,
                ):
                    yield result_bindings, proof + [raw]

    def _prove_all_candidates(
        self,
        goals: List[Expression],
        bindings: Bindings,
        steps: int,
        trail: set[str],
    ) -> Iterator[Tuple[Bindings, List[str]]]:
        if not goals:
            yield bindings, []
            return

        first = self._substitute(goals[0], bindings)
        for next_bindings, first_proof in self._prove_candidates(
            first,
            bindings,
            steps,
            trail,
        ):
            for final_bindings, rest_proof in self._prove_all_candidates(
                goals[1:],
                next_bindings,
                steps + 1,
                trail,
            ):
                yield final_bindings, first_proof + rest_proof

    def _unify(
        self,
        left: Expression,
        right: Expression,
        bindings: Bindings,
    ) -> Optional[Bindings]:
        left = self._resolve(left, bindings)
        right = self._resolve(right, bindings)

        if self._is_variable(left):
            bindings[left] = right
            return bindings
        if self._is_variable(right):
            bindings[right] = left
            return bindings
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                return None
            for left_item, right_item in zip(left, right):
                bindings = self._unify(left_item, right_item, bindings)
                if bindings is None:
                    return None
            return bindings
        if isinstance(left, str) and isinstance(right, str):
            if equivalent_symbol(left, right):
                return bindings
            pair = frozenset((canonical_symbol(left), canonical_symbol(right)))
            return bindings if pair in self._additional_equivalences else None
        return bindings if left == right else None

    def _resolve(self, value: Expression, bindings: Bindings) -> Expression:
        seen: set[str] = set()
        while self._is_variable(value) and value in bindings and value not in seen:
            seen.add(value)
            value = bindings[value]
        return value

    def _substitute(self, value: Expression, bindings: Bindings) -> Expression:
        value = self._resolve(value, bindings)
        if isinstance(value, list):
            return [self._substitute(item, bindings) for item in value]
        return value

    def _freshen(self, value: Expression, suffix: str) -> Expression:
        if self._is_variable(value):
            return f"{value}{suffix}"
        if isinstance(value, list):
            return [self._freshen(item, suffix) for item in value]
        return value

    def _is_variable(self, value: Expression) -> bool:
        return isinstance(value, str) and value.startswith(("$", "?"))

    def _render(self, value: Expression) -> str:
        if isinstance(value, list):
            return f"({' '.join(self._render(item) for item in value)})"
        return str(value)

    def _dedupe(self, statements: List[str]) -> List[str]:
        seen: set[str] = set()
        result: List[str] = []
        for statement in statements:
            if statement not in seen:
                seen.add(statement)
                result.append(statement)
        return result
