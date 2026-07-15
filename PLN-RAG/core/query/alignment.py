from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable


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
GENERIC_INTENT_TERMS = {
    "at",
    "become",
    "becoming",
    "classification",
    "classified",
    "classify",
    "considered",
    "elevated",
    "eligible",
    "eligibility",
    "high",
    "higher",
    "low",
    "lower",
    "qualifies",
    "qualified",
    "qualify",
    "risk",
    "status",
}
RISK_REDUCTION_TERMS = {
    "decreas",
    "decrease",
    "decreased",
    "decreases",
    "lower",
    "lowered",
    "lowers",
    "mitigat",
    "mitigate",
    "mitigated",
    "mitigates",
    "prevent",
    "prevented",
    "prevents",
    "protect",
    "protected",
    "protective",
    "protects",
    "reduc",
    "reduce",
    "reduced",
    "reduces",
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
    question_anchors = _question_anchor_terms(question_terms, question_entities)
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
                    question_anchors=question_anchors,
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
    """
    Keep only answer targets that match the user's requested predicate/topic.

    Qdrant retrieval is deliberately broad so it can supply supporting facts.
    The final executable target should be narrower: a provable intermediate fact
    should not answer a different question just because it came from the same
    retrieved chunk.
    """
    question_entities = extract_question_entities(question)
    question_terms = extract_question_terms(question)
    question_anchors = _question_anchor_terms(question_terms, question_entities)
    if not question_anchors:
        return _dedupe_queries(queries)

    filtered: list[str] = []
    seen: set[str] = set()
    for query in queries:
        clean = " ".join(str(query).split())
        if not clean or clean in seen:
            continue
        target = _query_target_expr(clean)
        if target is None:
            continue
        if _target_matches_question_intent(
            target,
            question_anchors=question_anchors,
            question_terms=question_terms,
            question_entities=question_entities,
        ):
            seen.add(clean)
            filtered.append(clean)
    return filtered


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
    return _expand_terms(terms)


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
    question_anchors: set[str],
    is_yes_no: bool,
) -> SExpr | None:
    if not _is_query_target(expr):
        return None

    variables = sorted(_variables(expr))

    # Grounded facts: query is fully grounded with question entities
    if not variables:
        constants = _constants(expr)
        if question_entities and not set(question_entities).issubset(constants):
            return None
        if question_terms and constants and not question_terms.intersection(constants):
            head_terms = _predicate_terms(expr)
            if not question_terms.intersection(head_terms):
                return None
        if not _target_matches_question_intent(
            expr,
            question_anchors=question_anchors,
            question_terms=question_terms,
            question_entities=question_entities,
        ):
            return None
        return expr

    # Partial grounding: single variable + question entity → bind
    if len(variables) == 1 and len(question_entities) == 1:
        grounded = _replace_symbol(expr, variables[0], question_entities[0])
        if not _grounded_contains_entities(grounded, question_entities):
            return None
        if not _target_matches_question_intent(
            grounded,
            question_anchors=question_anchors,
            question_terms=question_terms,
            question_entities=question_entities,
        ):
            return None
        return grounded

    # Yes/no question with multiple variables but has question entities to ground
    if is_yes_no and question_entities and variables:
        # Ground as many variables as possible with question entities
        result = expr
        remaining_entities = list(question_entities)
        for var in variables:
            if remaining_entities:
                result = _replace_symbol(result, var, remaining_entities[0])
                remaining_entities = remaining_entities[1:]
            else:
                break
        if not _constants(result):
            return None
        if not _grounded_contains_entities(result, question_entities):
            return None
        if not _target_matches_question_intent(
            result,
            question_anchors=question_anchors,
            question_terms=question_terms,
            question_entities=question_entities,
        ):
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


def _predicate_terms(expr: SExpr) -> set[str]:
    if not isinstance(expr, list) or not expr or not isinstance(expr[0], str):
        return set()
    head = _canonical_symbol(expr[0])
    return _expand_terms({part for part in head.split("_") if part})


def _target_terms(expr: SExpr, question_entities: list[str]) -> set[str]:
    constants = _constants(expr) - set(question_entities)
    return _predicate_terms(expr) | constants


def _target_matches_question_intent(
    expr: SExpr,
    *,
    question_anchors: set[str],
    question_terms: set[str],
    question_entities: list[str],
) -> bool:
    target_terms = _target_terms(expr, question_entities)
    if (
        target_terms.intersection(RISK_REDUCTION_TERMS)
        and not question_terms.intersection(RISK_REDUCTION_TERMS)
    ):
        return False
    if not question_anchors:
        return True
    return _covers_required_terms(question_anchors, target_terms)


def _question_anchor_terms(
    question_terms: set[str],
    question_entities: list[str],
) -> set[str]:
    entity_terms = set(question_entities)
    for entity in question_entities:
        entity_terms.update(part for part in entity.split("_") if part)
    anchors = set(question_terms)
    anchors.difference_update(entity_terms)
    anchors.difference_update(GENERIC_INTENT_TERMS)
    return anchors or (set(question_terms) - entity_terms)


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


def _score_grounded_target(
    expr: SExpr,
    *,
    question_terms: set[str],
    question_entities: list[str],
    qdrant_score: float,
    match_index: int,
) -> float:
    constants = _constants(expr)
    head_terms = _predicate_terms(expr)
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


def _expand_terms(terms: set[str]) -> set[str]:
    expanded = set(terms)
    for term in list(terms):
        if "_" in term:
            expanded.update(part for part in term.split("_") if part and part not in STOPWORDS)
        if term.endswith("ing") and len(term) > 5:
            base = term[:-3]
            expanded.add(base)
            expanded.add(base + "e")
        if term.endswith("ed") and len(term) > 4:
            expanded.add(term[:-2])
        if term in {"carb", "carbs"}:
            expanded.add("carbohydrate")
        if term == "carbohydrate":
            expanded.add("carb")
        if term == "obese":
            expanded.add("obesity")
        if term == "obesity":
            expanded.add("obese")
        if term in {"high", "higher", "excessive"}:
            expanded.update({"high", "higher", "excessive"})
        if term in {"waived", "waiv"}:
            expanded.add("waive")
        if term in {"qualified", "qualifies"}:
            expanded.add("qualify")
        if term in {"triggered", "trigger"}:
            expanded.update({"trigger", "triggered"})
    return expanded


def _covers_required_terms(required_terms: set[str], target_terms: set[str]) -> bool:
    normalized_target: set[str] = set()
    for term in target_terms:
        normalized_target.update(_term_alternates(term))
    for term in required_terms:
        if not _term_alternates(term).intersection(normalized_target):
            return False
    return True


def _term_alternates(term: str) -> set[str]:
    alternates = {term}
    if "_" in term:
        alternates.update(part for part in term.split("_") if part and part not in STOPWORDS)
    if term.endswith("ing") and len(term) > 5:
        base = term[:-3]
        alternates.update({base, base + "e"})
    if term.endswith("ed") and len(term) > 4:
        alternates.add(term[:-2])
        if term[:-1].endswith("e"):
            alternates.add(term[:-1])
    groups = [
        {"consume", "consumes", "consuming", "consumed", "consum"},
        {"high", "higher", "excessive"},
        {"obese", "obesity"},
        {"qualify", "qualifies", "qualified", "eligible", "eligibility"},
        {"classify", "classified", "classification", "classifi"},
        {"deny", "denied", "deni"},
        {"waive", "waived", "waiv"},
        {"lead", "led", "leading"},
        {"trigger", "triggered", "triggering", "triggers"},
        {"pollute", "polluted", "pollution"},
        {"reject", "rejected", "rejecting"},
        {"authorize", "authorized", "authorization"},
    ]
    for group in groups:
        if term in group:
            alternates.update(group)
    return alternates


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


def _canonical_symbol(token: str, lemmatize: bool = True) -> str:
    token = token.strip()
    if not token:
        return token
    token = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", token)
    token = token.replace("-", "_")
    token = re.sub(r"[^A-Za-z0-9_]", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    token = token.lower()
    if lemmatize:
        token = "_".join(_singularize(part) for part in token.split("_") if part)
    return token


def _singularize(word: str) -> str:
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ses") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word
