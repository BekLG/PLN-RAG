"""
Structural question analysis: what *kind* of question is this, and is a candidate
query well-formed?

Deciding whether a candidate target is what the question asks about is a semantic
judgment and lives in `core/query/target_gate.py`. This module used to make that
judgment with hardcoded English — synonym groups for `obesity`/`obese`,
`carb`/`carbohydrate`, `waive`/`waived`, plus term sets named `OPPOSITE_TERMS` and
`STATUS_QUALIFIER_TERMS` — and required every question content word to appear in
the target. That passed the bundled benchmark and rejected correct targets in every
other domain, so it is gone.

What remains is genuinely structural: yes/no vs. open vs. factors mode detection,
and `parse_query_signature`.

`direction`, `causal`, and `terms` survive as inputs to the separate
`Reasoner.explain_requirements` feature, which answers "which factors matter"
questions from rule premises rather than by proving a target.
"""

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
# Used only to derive `direction` and `causal` for explain_requirements, never to
# admit or reject a proof target.
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


def parse_question_intent(question: str) -> QuestionIntent:
    normalized = " ".join(str(question).strip().lower().split())
    tokens = re.findall(r"\b[A-Za-z0-9][A-Za-z0-9_-]*\b", normalized)
    terms = frozenset(_content_terms(tokens))
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


@dataclass(frozen=True)
class QueryIntentVerdict:
    """Why a candidate query was accepted or rejected by the structural gate."""

    accepted: bool
    stage: str = ""
    detail: str = ""

    def __bool__(self) -> bool:
        return self.accepted


_ACCEPTED = QueryIntentVerdict(accepted=True)


def evaluate_query_intent(intent: QuestionIntent, query: str) -> QueryIntentVerdict:
    """
    Structural admissibility only: is this a well-formed atomic proof target, and
    does the question mode execute one at all?

    Topical fit is the semantic gate's job. Keeping the two separate is what makes
    a rejection explainable — a structural rejection is always a real defect in the
    candidate, never a vocabulary mismatch.
    """
    if intent.mode in {QuestionMode.FACTORS, QuestionMode.SUFFICIENCY}:
        return QueryIntentVerdict(
            False,
            "intent_mode",
            f"{intent.mode.value} questions do not execute atomic proof targets",
        )
    if not parse_query_signature(query):
        return QueryIntentVerdict(
            False,
            "malformed",
            "query is not a (: $prf (Head args) $tv) form",
        )
    return _ACCEPTED


def query_matches_intent(intent: QuestionIntent, query: str) -> bool:
    return evaluate_query_intent(intent, query).accepted


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


def _content_terms(tokens: list[str]) -> set[str]:
    """
    Canonical content terms with light, language-general morphology.

    Previously this also applied a table of benchmark-specific synonyms
    (`carb`->`carbohydrate`, `obese`->`obesity`, `waived`->`waive`). Those made
    the bundled cases pass and generalized to nothing, so only the mechanical
    variants remain.
    """
    result: set[str] = set()
    for token in tokens:
        if token in STOPWORDS:
            continue
        term = canonical_symbol(token)
        if not term:
            continue
        result.add(term)
        if "_" in term:
            result.update(
                part for part in term.split("_") if part and part not in STOPWORDS
            )
        if term.endswith("ing") and len(term) > 5:
            result.add(term[:-3])
            result.add(term[:-3] + "e")
        if term.endswith("ed") and len(term) > 4:
            result.add(term[:-2])
    return result
