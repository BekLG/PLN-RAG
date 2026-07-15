from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from core.pln.symbol_normalization import canonical_symbol


class QuestionMode(str, Enum):
    BOOLEAN = "boolean"
    OPEN = "open"
    FACTORS = "factors"
    EXPLANATION = "explanation"
    SUFFICIENCY = "sufficiency"


@dataclass(frozen=True)
class QuestionIntent:
    mode: QuestionMode
    entities: tuple[str, ...]
    terms: frozenset[str]
    direction: str = ""
    causal: bool = False


YES_NO_STARTERS = {
    "is", "are", "was", "were", "does", "do", "did", "can", "could",
    "has", "have", "had", "may", "might", "must", "shall", "should",
}
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "does", "do",
    "did", "for", "from", "had", "has", "have", "how", "in", "is", "it", "of",
    "on", "or", "the", "this", "to", "was", "were", "what", "when",
    "where", "which", "who", "why", "with", "can", "could", "may",
    "might", "must", "shall", "should",
}
GENERIC_QUERY_TERMS = {
    "alone", "become", "becoming", "current", "currently", "factor",
    "risk", "sufficient", "sufficiency", "may", "might", "must", "shall",
    "should", "qualify", "qualified", "qualifies", "eligible", "eligibility",
    "classification", "classified", "classify", "considered", "status",
}
STATUS_QUALIFIER_TERMS = {
    "clinically", "diagnosed", "diagnosis", "evidence", "known", "reported",
}
CAUSAL_TERMS = {
    "cause", "causes", "caused", "causing", "contribute", "contributes",
    "lead", "leads", "leading", "result", "results", "sufficient",
}
INCREASE_TERMS = {
    "cause", "causes", "contribute", "contributes", "increase", "increases",
    "increasing", "lead", "leads", "raise", "raises",
}
REDUCE_TERMS = {
    "decrease", "decreases", "lower", "lowers", "mitigate", "mitigates",
    "prevent", "prevents", "protect", "protects", "reduce", "reduces",
}
OPPOSITE_TERMS = {
    "deny", "denied", "deni", "forbid", "forbidden", "invalid",
    "expired", "prevent", "prevented", "prohibit", "prohibited",
}


def parse_question_intent(question: str) -> QuestionIntent:
    normalized = " ".join(str(question).strip().lower().split())
    tokens = re.findall(r"\b[A-Za-z0-9][A-Za-z0-9_-]*\b", normalized)
    terms = frozenset(_expanded_terms(tokens))
    entities = tuple(_question_entities(question))

    has_factor_request = bool({"factor", "factors", "reason", "reasons"} & set(tokens))
    has_sufficiency = (
        bool({"sufficient", "enough", "alone"} & set(tokens))
        and bool(CAUSAL_TERMS & terms)
    )
    if has_sufficiency:
        mode = QuestionMode.SUFFICIENCY
    elif has_factor_request:
        mode = QuestionMode.FACTORS
    elif tokens and tokens[0] == "why":
        mode = QuestionMode.EXPLANATION
    elif tokens and tokens[0] in YES_NO_STARTERS:
        mode = QuestionMode.BOOLEAN
    else:
        mode = QuestionMode.OPEN

    direction = ""
    if terms & REDUCE_TERMS:
        direction = "reduce"
    elif terms & INCREASE_TERMS:
        direction = "increase"

    return QuestionIntent(
        mode=mode,
        entities=entities,
        terms=terms,
        direction=direction,
        causal=bool(terms & CAUSAL_TERMS),
    )


def query_matches_intent(intent: QuestionIntent, query: str) -> bool:
    """Proof gate for executable atomic queries."""
    if intent.mode in {QuestionMode.FACTORS, QuestionMode.SUFFICIENCY}:
        return False

    signature = parse_query_signature(query)
    if not signature:
        return False
    if intent.entities and not all(
        _entity_is_represented(entity, signature["args"])
        for entity in intent.entities
    ):
        return False

    predicate_terms = _predicate_terms(signature["head"])
    if predicate_terms & OPPOSITE_TERMS and not set(intent.terms) & OPPOSITE_TERMS:
        return False
    if predicate_terms.intersection(STATUS_QUALIFIER_TERMS):
        question_status_terms = set(intent.terms).intersection(STATUS_QUALIFIER_TERMS)
        if not question_status_terms:
            return False

    target_terms = set(predicate_terms)
    for arg in signature["args"]:
        canonical = canonical_symbol(str(arg))
        if canonical in set(intent.entities):
            continue
        target_terms.update(_expanded_terms(canonical.split("_")))
    required_terms = _required_content_terms(intent)
    if not required_terms:
        return True

    if intent.direction == "reduce" and not predicate_terms.intersection(REDUCE_TERMS):
        return False
    if intent.causal and not predicate_terms.intersection(
        CAUSAL_TERMS | INCREASE_TERMS | {"risk", "obesity", "diabetes"}
    ):
        return False
    return _covers_required_terms(required_terms, target_terms)


def query_intent_score(intent: QuestionIntent, query: str) -> int:
    signature = parse_query_signature(query)
    if not signature:
        return 0
    predicate_terms = _predicate_terms(signature["head"])
    arg_terms: set[str] = set()
    for arg in signature["args"]:
        canonical = canonical_symbol(str(arg))
        if canonical in set(intent.entities):
            continue
        arg_terms.update(_expanded_terms(canonical.split("_")))
    content_terms = _required_content_terms(intent)
    return (
        3 * len(predicate_terms.intersection(content_terms))
        + len(arg_terms.intersection(content_terms))
    )


def parse_query_signature(query: str) -> dict[str, object] | None:
    clean = " ".join(str(query).split())
    match = re.fullmatch(
        r"\(:\s+[$?][^\s]+\s+\(([A-Za-z][A-Za-z0-9_]*)((?:\s+[^()\s]+)+)\)\s+[$?][^\s]+\)",
        clean,
    )
    if not match:
        return None
    args = [canonical_symbol(arg) for arg in match.group(2).split()]
    return {"head": match.group(1), "args": args, "arity": len(args)}


def _question_entities(question: str) -> list[str]:
    entities: list[str] = []
    for token in re.findall(r"\b[A-Z][A-Za-z0-9_-]*\b", question):
        if token.lower() in STOPWORDS or token.lower() in YES_NO_STARTERS:
            continue
        entity = canonical_symbol(token)
        if entity and entity not in entities:
            entities.append(entity)
    return entities


def _expanded_terms(tokens: list[str]) -> set[str]:
    result: set[str] = set()
    for token in tokens:
        if token in STOPWORDS:
            continue
        term = canonical_symbol(token)
        if not term:
            continue
        result.add(term)
        if "_" in term:
            result.update(part for part in term.split("_") if part and part not in STOPWORDS)
        if term.endswith("ing") and len(term) > 5:
            result.add(term[:-3])
            result.add(term[:-3] + "e")
        if term.endswith("ed") and len(term) > 4:
            result.add(term[:-2])
        if term == "led":
            result.add("lead")
        if term in {"denied", "deni"}:
            result.add("deny")
        if term == "automatically":
            result.add("automatic")
        if term in {"carb", "carbs"}:
            result.add("carbohydrate")
        if term == "carbohydrate":
            result.add("carb")
        if term == "obese":
            result.add("obesity")
        if term == "obesity":
            result.add("obese")
        if term in {"high", "higher", "excessive"}:
            result.update({"high", "higher", "excessive"})
        if term in {"waived", "waiv"}:
            result.add("waive")
        if term in {"qualified", "qualifies"}:
            result.add("qualify")
        if term in {"triggered", "trigger"}:
            result.update({"trigger", "triggered"})
    return result


def _entity_is_represented(entity: str, args: object) -> bool:
    canonical_entity = canonical_symbol(entity)
    for arg in args if isinstance(args, list) else []:
        canonical_arg = canonical_symbol(str(arg))
        if canonical_arg == canonical_entity:
            return True
        if canonical_entity in canonical_arg.split("_"):
            return True
    return False


def _predicate_terms(predicate: str) -> set[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", predicate)
    return _expanded_terms(
        [part.lower() for part in re.split(r"[^A-Za-z0-9]+", spaced) if part]
    )


def _required_content_terms(intent: QuestionIntent) -> set[str]:
    entity_terms = set(intent.entities)
    for entity in intent.entities:
        entity_terms.update(part for part in entity.split("_") if part)
    return set(intent.terms) - entity_terms - GENERIC_QUERY_TERMS


def _covers_required_terms(required_terms: set[str], target_terms: set[str]) -> bool:
    if not required_terms:
        return True
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
