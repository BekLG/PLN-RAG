from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from core.pln.symbol_normalization import canonical_symbol


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "do",
    "does",
    "did",
    "for",
    "from",
    "has",
    "have",
    "had",
    "how",
    "in",
    "is",
    "it",
    "may",
    "might",
    "must",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "shall",
    "should",
}
YES_NO_STARTERS = {
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
}
NON_QUERY_TARGET_HEADS = {
    "Implication",
    "Premises",
    "Conclusions",
    "STV",
    "And",
    "Or",
}

SExpr = str | list["SExpr"]


@dataclass
class QueryAlignmentResult:
    queries: list[str] = field(default_factory=list)
    matches: list[dict[str, Any]] = field(default_factory=list)


def extract_query_targets(statements: Iterable[str]) -> list[str]:
    """Extract fact payloads and rule conclusions that can become queries."""
    targets: list[str] = []
    seen: set[str] = set()

    for statement in statements:
        for expr in _parse_sexprs(statement):
            for target in _targets_from_statement(expr):
                serialized = _serialize(target)
                if serialized and serialized not in seen:
                    seen.add(serialized)
                    targets.append(serialized)

    return targets


def build_aligned_queries(
    question: str,
    matches: Iterable[dict[str, Any]],
) -> QueryAlignmentResult:
    question_entities = extract_question_entities(question)
    question_terms = extract_question_terms(question)
    is_yes_no = _is_yes_no_question(question)

    ranked_queries: list[tuple[float, str]] = []
    seen_queries: set[str] = set()
    debug_matches: list[dict[str, Any]] = []

    for match_index, match in enumerate(matches):
        targets = _targets_from_match(match)
        match_queries: list[str] = []

        for target in targets:
            for expr in _parse_sexprs(target):
                grounded = _ground_target(
                    expr,
                    question_entities=question_entities,
                    question_terms=question_terms,
                    is_yes_no=is_yes_no,
                )
                if grounded is None:
                    continue
                query = f"(: $prf {_serialize(grounded)} $tv)"
                if query in seen_queries:
                    continue
                seen_queries.add(query)
                ranked_queries.append(
                    (
                        _score_grounded_target(
                            grounded,
                            question_terms=question_terms,
                            question_entities=question_entities,
                            qdrant_score=float(match.get("score") or 0.0),
                            match_index=match_index,
                        ),
                        query,
                    )
                )
                match_queries.append(query)

        debug_matches.append(
            {
                "score": match.get("score"),
                "nl": match.get("nl", ""),
                "pln": match.get("pln", []),
                "query_targets": targets,
                "aligned_queries": match_queries,
            }
        )

    ranked_queries.sort(key=lambda item: item[0], reverse=True)
    return QueryAlignmentResult(
        queries=[query for _, query in ranked_queries],
        matches=debug_matches,
    )


def filter_queries_by_question_intent(question: str, queries: Iterable[str]) -> list[str]:
    """Structural filter over retrieved targets. Topical fit is the gate's job."""
    return filter_queries_by_question_intent_verbose(question, queries)[0]


def filter_queries_by_question_intent_verbose(
    question: str,
    queries: Iterable[str],
) -> tuple[list[str], list[tuple[str, str, str]]]:
    """
    Drop only malformed targets, returning (kept, rejections).

    This used to be a topical filter: it required every question anchor term to
    appear in the target's predicate or arguments, using a hardcoded synonym table.
    Because it ran *before* the service-level gate, anything it discarded was
    invisible to every later diagnostic — and it discarded correct targets, since
    `recalibrated` never matched `recalibration`. Deciding what the question is
    about now happens once, semantically, in `core/query/target_gate.py`.

    Retrieval stays deliberately broad so it can also supply supporting facts;
    narrowing to the answer target is the gate's responsibility.
    """
    kept: list[str] = []
    rejected: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for query in queries:
        clean = " ".join(str(query).split())
        if not clean or clean in seen:
            continue
        if _query_target_expr(clean) is None:
            rejected.append((clean, "malformed", "not a proof-target s-expression"))
            continue
        seen.add(clean)
        kept.append(clean)
    return kept, rejected


def extract_forward_seed_terms(matches: Iterable[dict[str, Any]]) -> list[str]:
    """Extract grounded fact bodies from Qdrant matches for PeTTa forward chaining."""
    seeds: list[str] = []
    seen: set[str] = set()

    for match in matches:
        pln = match.get("pln") or []
        if not isinstance(pln, list):
            continue
        for statement in pln:
            for expr in _parse_sexprs(str(statement)):
                payload = _statement_payload(expr)
                if payload is None:
                    continue
                if _payload_is_rule(payload) or _variables(payload):
                    continue
                serialized = _serialize(payload)
                if serialized and serialized not in seen:
                    seen.add(serialized)
                    seeds.append(serialized)

    return seeds


def extract_question_entities(question: str) -> list[str]:
    entities: list[str] = []
    for token in re.findall(r"\b[A-Z][A-Za-z0-9_-]*\b", question):
        if token.lower() in STOPWORDS or token.lower() in YES_NO_STARTERS:
            continue
        canonical = _canonical_symbol(token, lemmatize=False)
        if canonical and canonical not in entities:
            entities.append(canonical)
    return entities


def extract_question_terms(question: str) -> set[str]:
    terms: set[str] = set()
    for token in re.findall(r"\b[A-Za-z0-9][A-Za-z0-9_-]*\b", question):
        lowered = token.lower()
        if len(lowered) < 2 or lowered in STOPWORDS:
            continue
        canonical = _canonical_symbol(lowered)
        if canonical:
            terms.add(canonical)
    return terms


def _targets_from_match(match: dict[str, Any]) -> list[str]:
    targets = match.get("query_targets") or []
    if isinstance(targets, list) and targets:
        return [str(target) for target in targets if str(target).strip()]

    pln = match.get("pln") or []
    if isinstance(pln, list):
        return extract_query_targets([str(item) for item in pln])
    return []


def _targets_from_statement(expr: SExpr) -> list[SExpr]:
    payload = _statement_payload(expr)
    if payload is None:
        return []

    if _payload_is_rule(payload):
        return [
            conclusion
            for conclusion in _extract_conclusions(payload)
            if _is_query_target(conclusion)
        ]

    return [payload] if _is_query_target(payload) else []


def _extract_conclusions(expr: list[SExpr]) -> list[SExpr]:
    conclusions: list[SExpr] = []
    for item in expr:
        if (
            isinstance(item, list)
            and item
            and item[0] == "Conclusions"
        ):
            conclusions.extend(child for child in item[1:] if isinstance(child, list))
    return conclusions


def _is_query_target(expr: SExpr) -> bool:
    return (
        isinstance(expr, list)
        and bool(expr)
        and isinstance(expr[0], str)
        and expr[0] not in NON_QUERY_TARGET_HEADS
    )


def _ground_target(
    expr: SExpr,
    *,
    question_entities: list[str],
    question_terms: set[str],
    is_yes_no: bool,
) -> SExpr | None:
    """
    Ground a retrieved target with entities named in the question.

    Purely structural: can this target be grounded from what the question
    mentions? The topical vetoes that used to live at each return point were term
    overlap checks against a hardcoded synonym table, and they rejected correct
    targets. Producing a groundable candidate here and letting the semantic gate
    judge it keeps one decision in one place.

    `question_terms` is retained because `build_aligned_queries` still uses it to
    rank candidates; ranking is advisory, unlike the vetoes that were removed.
    """
    if not _is_query_target(expr):
        return None

    variables = sorted(_variables(expr))

    # Fully grounded target: usable when it mentions every question entity.
    if not variables:
        if question_entities and not set(question_entities).issubset(_constants(expr)):
            return None
        return expr

    # Single variable and a single question entity: bind them.
    if len(variables) == 1 and len(question_entities) == 1:
        grounded = _replace_symbol(expr, variables[0], question_entities[0])
        if not _grounded_contains_entities(grounded, question_entities):
            return None
        return grounded

    # Yes/no with several variables: bind as many as the question supplies.
    if is_yes_no and question_entities and variables:
        result = expr
        remaining_entities = list(question_entities)
        for var in variables:
            if not remaining_entities:
                break
            result = _replace_symbol(result, var, remaining_entities[0])
            remaining_entities = remaining_entities[1:]
        if not _constants(result):
            return None
        if not _grounded_contains_entities(result, question_entities):
            return None
        return result

    return None


def _grounded_contains_entities(expr: SExpr, question_entities: list[str]) -> bool:
    if not question_entities:
        return True
    return set(question_entities).issubset(_constants(expr))


def _variables(expr: SExpr) -> set[str]:
    if isinstance(expr, str):
        return {expr} if expr.startswith(("$", "?")) and expr not in {"$prf", "$tv", "?prf", "?tv"} else set()
    variables: set[str] = set()
    for item in expr:
        variables.update(_variables(item))
    return variables


def _constants(expr: SExpr, *, is_head: bool = True) -> set[str]:
    if isinstance(expr, str):
        if is_head or expr.startswith(("$", "?")):
            return set()
        canonical = _canonical_symbol(expr)
        return {canonical} if canonical and not canonical.replace("_", "").isdigit() else set()

    constants: set[str] = set()
    for index, item in enumerate(expr):
        constants.update(_constants(item, is_head=index == 0))
    return constants


def _statement_payload(expr: SExpr) -> SExpr | None:
    if not isinstance(expr, list) or len(expr) < 3 or expr[0] != ":":
        return None
    payload = expr[2]
    if not isinstance(payload, list) or not payload:
        return None
    return payload


def _payload_is_rule(expr: SExpr) -> bool:
    return isinstance(expr, list) and bool(expr) and expr[0] == "Implication"


def _query_target_expr(query: str) -> SExpr | None:
    roots = _parse_sexprs(query)
    if len(roots) != 1:
        return None
    expr = roots[0]
    if not isinstance(expr, list) or len(expr) < 3 or expr[0] != ":":
        return None
    target = expr[2]
    return target if _is_query_target(target) else None


def _dedupe_queries(queries: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for query in queries:
        clean = " ".join(str(query).split())
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result


def _head_terms(expr: SExpr) -> set[str]:
    """Canonical parts of a target's predicate name, for ranking only."""
    if not isinstance(expr, list) or not expr or not isinstance(expr[0], str):
        return set()
    head = _canonical_symbol(expr[0])
    return {part for part in head.split("_") if part}


def _score_grounded_target(
    expr: SExpr,
    *,
    question_terms: set[str],
    question_entities: list[str],
    qdrant_score: float,
    match_index: int,
) -> float:
    constants = _constants(expr)
    head_terms = _head_terms(expr)
    entity_fit = len(set(question_entities).intersection(constants))
    head_overlap = len(question_terms.intersection(head_terms))
    constant_overlap = len(question_terms.intersection(constants))
    head = expr[0] if isinstance(expr, list) and expr else ""
    structural_penalty = 12.0 if head in {"IsA"} else 0.0
    return (
        qdrant_score * 100.0
        + entity_fit * 40.0
        + head_overlap * 24.0
        + constant_overlap * 6.0
        - structural_penalty
        - match_index * 0.01
    )


def _replace_symbol(expr: SExpr, old: str, new: str) -> SExpr:
    if isinstance(expr, str):
        return new if expr == old else expr
    return [_replace_symbol(item, old, new) for item in expr]


def _is_yes_no_question(question: str) -> bool:
    tokens = re.findall(r"\b[A-Za-z]+\b", question.lower())
    return bool(tokens) and tokens[0] in YES_NO_STARTERS


def _parse_sexprs(text: str) -> list[SExpr]:
    tokens = re.findall(r"\(|\)|[^\s()]+", text)
    roots: list[SExpr] = []
    stack: list[list[SExpr]] = []

    for token in tokens:
        if token == "(":
            node: list[SExpr] = []
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


def _serialize(expr: SExpr) -> str:
    if isinstance(expr, str):
        return expr
    return "(" + " ".join(_serialize(item) for item in expr) + ")"


# Canonicalization lives in core.pln.symbol_normalization so ingestion and query
# planning cannot drift apart. This module previously carried its own copy that
# disagreed on invariant plurals (`series` -> `sery`).
def _canonical_symbol(token: str, lemmatize: bool = True) -> str:
    return canonical_symbol(token, lemmatize=lemmatize)
