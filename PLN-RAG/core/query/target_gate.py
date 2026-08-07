"""
Semantic gate deciding which proof targets a question actually asks about.

This replaces the hardcoded term-matching layer that used to live in
`core/query/intent.py` and `core/query/alignment.py`. That layer required every
content word of the question to appear in the target's predicate or arguments,
backed by literal synonym tables (`obesity`/`obese`, `carb`/`carbohydrate`,
`waive`/`waived`). It passed the bundled benchmark and rejected correct targets
everywhere else: `recalibrated` never matched `recalibration`, and a question with
a trailing clause produced ten required terms against a three-term predicate.

Two stages:

1. **Embedding rank** — cheap, drops nothing. Each candidate is glossed to natural
   language and embedded with the `nomic-embed-text` instance already serving the
   vector store, then scored by cosine against the question.
2. **LLM verdict** — one batched, schema-constrained call over the top `top_k`
   candidates, returning `asks_exactly` / `asks_negation` / `related` / `unrelated`.

The gate only decides which targets are *worth attempting a proof for*. It never
asserts truth: the reasoner still decides that, so a wrong admission costs an
`unknown`, not a wrong answer. When the gate is unavailable it falls back to the
embedding ranking with no veto rather than going mute, and records that it did.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from core.pln.predicate_mapping import humanize_predicate
from core.query.intent import parse_query_signature


ASKS_EXACTLY = "asks_exactly"
ASKS_NEGATION = "asks_negation"
EPISTEMIC_PROXY = "epistemic_proxy"
RELATED = "related"
UNRELATED = "unrelated"

TARGET_VERDICTS = frozenset(
    {ASKS_EXACTLY, ASKS_NEGATION, EPISTEMIC_PROXY, RELATED, UNRELATED}
)
ADMITTING_VERDICTS = frozenset({ASKS_EXACTLY, ASKS_NEGATION})

_GATE_PROMPT = """You decide which formal proof targets a question is asking about.

For each candidate below, choose exactly one verdict:
- asks_exactly: proving this target answers the question as asked.
- asks_negation: this target is the strict logical negation of the asked
  proposition. Proving it answers the question with "no".
- epistemic_proxy: the target is about the EVIDENCE, KNOWLEDGE, RECORD, or
  REPORTING status of the proposition rather than the proposition itself.
- related: same topic, but proving it would answer a different question.
- unrelated: not about what the question asks.

Judge meaning, not wording. "Must the turbine be recalibrated?" and
"needs recalibration: turbine" are asks_exactly despite different surface forms.
A target whose predicate lexicalizes negation, such as "not eligible for renewal",
is asks_negation for a question asking whether the entity IS eligible.

Separate the TOPIC of a question from its CIRCUMSTANCE. The topic is the property
being asked about; the circumstance is any subordinate clause saying when, why,
after what, or under what conditions. Match on topic and ignore circumstance: a
target that states the main clause is asks_exactly even when the question adds a
clause the target does not mention. For "Did the shuttle enter safe mode after
losing network contact?" the topic is entering safe mode, so a target meaning
"enters safe mode: shuttle" is asks_exactly — not related. Do not require the
circumstance to appear in the target.

epistemic_proxy is critical and easy to miss. For "May Omar access the secure
archive?", a target meaning "has evidence against archive access: omar" is
epistemic_proxy, NOT asks_negation: evidence against a claim is not the claim's
negation, and a missing or expired credential is not a proof of denial. The same
applies to targets meaning not known, not reported, not recorded, not diagnosed,
not classified, or no evidence found. Treating any of these as a negation turns an
absence of information into a confident "no".

Reserve asks_negation for a genuine logical opposite of the asked proposition.

Targets marked in_knowledge_base=yes use the vocabulary the knowledge base already
stores. This matters only for wording, never for topic. If such a target expresses
the asked proposition, do not downgrade it over a shade of tense or aspect:
"received contractor approval: omar" answers "Does Omar have contractor approval?"
and is asks_exactly, not related. has/received, completed/finished and is/becomes
are the same proposition.

Being stored does NOT make a target relevant. A target about a different property of
the same entity is still unrelated, however it is marked. For "Is the bolt of fabric
discarded?", a target meaning "is a: bolt of fabric, item" is unrelated even though
it is stored and trivially provable.

So: be strict about which property is asked about, forgiving about wording and about
circumstance clauses.

QUESTION: {question}

CANDIDATES:
{candidates}

Return JSON: {{"results":[{{"id":0,"verdict":"...","confidence":0.0,"reason":"..."}}]}}
"""


@dataclass(frozen=True)
class TargetVerdict:
    target: str
    verdict: str
    confidence: float
    reason: str
    embedding_score: float = 0.0
    source: str = "llm"

    @property
    def admits(self) -> bool:
        return self.verdict in ADMITTING_VERDICTS

    @property
    def wants_negative_execution(self) -> bool:
        return self.verdict == ASKS_NEGATION


@dataclass(frozen=True)
class GateDecision:
    """Admitted targets in rank order, plus a reason for every rejection."""

    admitted: tuple[str, ...]
    verdicts: dict[str, TargetVerdict]
    rejections: tuple[tuple[str, str, str], ...]  # (target, stage, detail)
    gate_available: bool


def target_gloss(query_or_target: str, card_lookup=None) -> str:
    """
    Render a proof target as natural language for embedding and prompting.

    `(RequiresRecalibration turbine)` becomes "requires recalibration: turbine".
    When a predicate card exists its label and definition are preferred, since
    cards are already built and embedded during ingestion.
    """
    signature = _signature_of(query_or_target)
    if not signature:
        return " ".join(str(query_or_target).split())

    head = str(signature["head"])
    negated = bool(signature.get("negated"))
    phrase = humanize_predicate(head)

    if card_lookup:
        try:
            card = card_lookup(head, int(signature["arity"]))
        except Exception:
            card = None
        if card:
            label = str(card.get("label") or "").strip()
            definition = str(card.get("definition") or "").strip()
            if label:
                phrase = label
            if definition:
                phrase = f"{phrase} ({definition})"

    raw_args = signature.get("raw_args") or signature["args"]
    args = ", ".join(
        "(unknown - to be found)"
        if str(raw).startswith(("$", "?"))
        else humanize_predicate(str(shown))
        for raw, shown in zip(raw_args, signature["args"])
        if str(shown).strip()
    )
    gloss = f"{phrase}: {args}" if args else phrase
    return f"not ({gloss})" if negated else gloss


class SemanticTargetGate:
    """Embedding rank plus one batched LLM verdict per question."""

    GEMINI_MIN_TIMEOUT_SECONDS = 10

    def __init__(
        self,
        *,
        enabled: bool,
        embedder=None,
        card_lookup=None,
        gemini_api_key: str | None = None,
        gemini_model: str = "gemini-2.5-flash",
        openai_api_key: str | None = None,
        openai_model: str = "gpt-4o-mini",
        top_k: int = 5,
        timeout_seconds: int = 12,
        min_confidence: float = 0.7,
        negation_min_confidence: float = 0.85,
        cache_max_entries: int = 256,
    ):
        self.enabled = enabled
        self._embedder = embedder
        self._card_lookup = card_lookup
        self.gemini_api_key = gemini_api_key
        self.gemini_model = gemini_model
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.top_k = max(1, int(top_k))
        self.timeout_seconds = max(1, int(timeout_seconds))
        self.min_confidence = float(min_confidence)
        # asks_negation inverts the reported answer, so a mistaken verdict becomes a
        # confident wrong answer rather than an `unknown`. Hold it to a higher bar
        # than a plain admission.
        self.negation_min_confidence = max(
            float(min_confidence), float(negation_min_confidence)
        )
        self._cache: dict[tuple[str, tuple[str, ...]], GateDecision] = {}
        self._cache_max_entries = max(1, int(cache_max_entries))

    @property
    def llm_available(self) -> bool:
        return bool(self.enabled and (self.gemini_api_key or self.openai_api_key))

    def decide(
        self,
        question: str,
        candidates: Sequence[str],
        kb_grounded: Sequence[str] | None = None,
    ) -> GateDecision:
        """
        Choose which candidates to attempt.

        `kb_grounded` names candidates built from atoms the knowledge base actually
        stores. Those are the only ones that can be proved, so they are surfaced to
        the model and preferred on ties. Without that signal the gate weighed a
        parser-invented predicate equally against the stored one and picked the
        unprovable one on embedding similarity alone.
        """
        clean = _dedupe([" ".join(str(item).split()) for item in candidates])
        if not clean:
            return GateDecision((), {}, (), True)
        if not self.enabled:
            return GateDecision(tuple(clean), {}, (), True)

        grounded = {" ".join(str(item).split()) for item in (kb_grounded or ())}
        key = (
            " ".join(str(question).split()),
            tuple(sorted(clean)),
            tuple(sorted(grounded)),
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        ranked = self._rank(question, clean)
        decision = self._verdicts(question, ranked, grounded)
        self._remember(key, decision)
        return decision

    def _rank(self, question: str, candidates: list[str]) -> list[tuple[float, str]]:
        """Cosine rank against the question. Returns every candidate, ordered."""
        if not self._embedder:
            return [(0.0, target) for target in candidates]
        glosses = [target_gloss(target, self._card_lookup) for target in candidates]
        try:
            vectors = self._embedder.embed_many([question, *glosses])
        except Exception as exc:
            print(f"[TargetGate] embedding rank unavailable; keeping order: {exc}")
            return [(0.0, target) for target in candidates]
        if len(vectors) != len(candidates) + 1:
            return [(0.0, target) for target in candidates]
        question_vector = vectors[0]
        scored = [
            (_cosine(question_vector, vector), target)
            for vector, target in zip(vectors[1:], candidates)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    def _verdicts(
        self,
        question: str,
        ranked: list[tuple[float, str]],
        grounded: set[str] | None = None,
    ) -> GateDecision:
        grounded = grounded or set()
        considered = ranked[: self.top_k]
        deferred = ranked[self.top_k :]

        if not self.llm_available:
            # No veto without a judge: keep the embedding order so the reasoner
            # still gets a chance, and say so rather than failing silently.
            admitted = tuple(target for _score, target in ranked)
            verdicts = {
                target: TargetVerdict(
                    target=target,
                    verdict=ASKS_EXACTLY,
                    confidence=0.0,
                    reason="semantic gate unavailable; admitted on embedding rank",
                    embedding_score=score,
                    source="embedding_fallback",
                )
                for score, target in ranked
            }
            return GateDecision(admitted, verdicts, (), False)

        rows = self._classify(question, considered, grounded)
        if rows is None:
            admitted = tuple(target for _score, target in ranked)
            verdicts = {
                target: TargetVerdict(
                    target=target,
                    verdict=ASKS_EXACTLY,
                    confidence=0.0,
                    reason="semantic gate call failed; admitted on embedding rank",
                    embedding_score=score,
                    source="embedding_fallback",
                )
                for score, target in ranked
            }
            return GateDecision(admitted, verdicts, (), False)

        admitted: list[str] = []
        verdicts: dict[str, TargetVerdict] = {}
        rejections: list[tuple[str, str, str]] = []

        for index, (score, target) in enumerate(considered):
            row = rows.get(index)
            if not row:
                # Unjudged inside the considered window: admit rather than veto,
                # because a missing row is a gate failure, not evidence.
                verdicts[target] = TargetVerdict(
                    target, ASKS_EXACTLY, 0.0,
                    "no verdict returned for this candidate; admitted",
                    score, "embedding_fallback",
                )
                admitted.append(target)
                continue

            verdict = str(row.get("verdict") or "").strip()
            if verdict not in TARGET_VERDICTS:
                verdict = RELATED
            confidence = _clamp(row.get("confidence"))
            reason = str(row.get("reason") or "").strip() or "model verdict"
            record = TargetVerdict(target, verdict, confidence, reason, score)
            verdicts[target] = record

            if not record.admits:
                rejections.append((
                    target,
                    "semantic_gate",
                    f"{verdict} (confidence {confidence:.2f}): {reason}",
                ))
                continue
            floor = (
                self.negation_min_confidence
                if verdict == ASKS_NEGATION
                else self.min_confidence
            )
            if confidence < floor:
                rejections.append((
                    target,
                    "semantic_gate_low_confidence",
                    f"{verdict} but confidence {confidence:.2f} "
                    f"< {floor:.2f}: {reason}",
                ))
                continue
            admitted.append(target)

        for score, target in deferred:
            rejections.append((
                target,
                "semantic_gate_not_considered",
                f"ranked below top {self.top_k} by embedding similarity "
                f"(score {score:.3f})",
            ))

        # Order: exact answers before negations, then stored vocabulary before
        # parser-invented, then embedding similarity. Grounding only breaks ties
        # among targets the gate judged equally responsive — promoting a stored
        # but off-topic target above a relevant one produced a trivially true
        # proof of the wrong proposition.
        def rank_key(target: str):
            record = verdicts[target]
            return (
                0 if record.verdict == ASKS_EXACTLY else 1,
                -round(record.confidence, 2),
                target not in grounded,
                -record.embedding_score,
            )

        admitted.sort(key=rank_key)
        return GateDecision(tuple(admitted), verdicts, tuple(rejections), True)

    def _classify(
        self,
        question: str,
        considered: list[tuple[float, str]],
        grounded: set[str] | None = None,
    ) -> dict[int, dict[str, Any]] | None:
        if not considered:
            return {}
        grounded = grounded or set()
        lines = []
        for index, (_score, target) in enumerate(considered):
            in_kb = "yes" if target in grounded else "no"
            lines.append(
                f"{index}. target={target} meaning=\""
                f"{target_gloss(target, self._card_lookup)}\" "
                f"in_knowledge_base={in_kb}"
            )
        prompt = _GATE_PROMPT.format(
            question=" ".join(str(question).split()),
            candidates="\n".join(lines),
        )
        try:
            if self.gemini_api_key:
                if self.timeout_seconds < self.GEMINI_MIN_TIMEOUT_SECONDS:
                    return None
                raw = self._call_gemini(prompt)
            else:
                raw = self._call_openai(prompt)
            payload = _parse_json_object(raw)
            results = payload.get("results", [])
            if not isinstance(results, list):
                return None
            return {
                int(row["id"]): row
                for row in results
                if isinstance(row, dict) and str(row.get("id", "")).lstrip("-").isdigit()
            }
        except Exception as exc:
            print(f"[TargetGate] verdict call failed: {exc}")
            return None

    def _call_gemini(self, prompt: str) -> Any:
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=self.gemini_api_key,
            http_options=types.HttpOptions(timeout=self.timeout_seconds * 1000),
        )
        response = client.models.generate_content(
            model=self.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=1200,
                response_mime_type="application/json",
                response_json_schema=_gate_response_schema(),
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        parsed = getattr(response, "parsed", None)
        return parsed if isinstance(parsed, dict) else str(response.text or "")

    def _call_openai(self, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(
            api_key=self.openai_api_key,
            timeout=float(self.timeout_seconds),
        )
        response = client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return strict JSON. Decide conservatively which proof "
                        "target a question asks about."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )
        return str(response.choices[0].message.content or "")

    def _remember(self, key, decision: GateDecision) -> None:
        if len(self._cache) >= self._cache_max_entries:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = decision


def _gate_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "verdict": {
                            "type": "string",
                            "enum": sorted(TARGET_VERDICTS),
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["id", "verdict", "confidence", "reason"],
                },
            }
        },
        "required": ["results"],
    }


def _signature_of(query_or_target: str) -> dict[str, Any] | None:
    """Accept either a full `(: $prf (Head args) $tv)` query or a bare target."""
    text = " ".join(str(query_or_target).split())
    signature = parse_query_signature(text)
    if signature:
        return {**signature, "negated": False, "raw_args": _raw_args(text)}

    inner = text
    negated = False
    match = re.fullmatch(r"\(:\s+[$?][^\s]+\s+(\(.+\))\s+[$?][^\s]+\)", text)
    if match:
        inner = match.group(1)
    negation = re.fullmatch(r"\(Not\s+(\(.+\))\)", inner)
    if negation:
        negated = True
        inner = negation.group(1)
    signature = parse_query_signature(f"(: $prf {inner} $tv)")
    if not signature:
        return None
    return {
        **signature,
        "negated": negated,
        "raw_args": _raw_args(f"(: $prf {inner} $tv)"),
    }


def _raw_args(query: str) -> list[str]:
    """Argument tokens exactly as written, so `$x` stays recognizable."""
    match = re.fullmatch(
        r"\(:\s+[$?][^\s]+\s+\(([A-Za-z][A-Za-z0-9_]*)((?:\s+[^()\s]+)+)\)\s+[$?][^\s]+\)",
        " ".join(str(query).split()),
    )
    return match.group(2).split() if match else []


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _clamp(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def _parse_json_object(text: Any) -> dict[str, Any]:
    if isinstance(text, dict):
        return text
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("target gate did not return a JSON object")
    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("target gate returned a non-object payload")
    return parsed


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result
