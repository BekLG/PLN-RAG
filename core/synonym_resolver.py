import json
import math
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import httpx
from openai import OpenAI

from config import get_settings
from core.horn_fallback import parse_expression
from core.symbol_normalization import canonical_symbol, equivalent_symbol


RELATIONS = {
    "same_meaning",
    "broader",
    "narrower",
    "related",
    "unrelated",
    "ambiguous",
}

IGNORED_SYMBOLS = {
    ":",
    "Implication",
    "Premises",
    "Conclusions",
    "And",
    "Or",
    "Not",
    "STV",
}


@dataclass(frozen=True)
class SynonymDecision:
    left: str
    right: str
    relation: str
    confidence: float
    source: str
    reason: str
    updated_at: str


class SynonymResolver:
    def __init__(
        self,
        settings: Any | None = None,
        http_client: httpx.Client | None = None,
        openai_client: OpenAI | None = None,
    ):
        self._settings = settings or get_settings()
        self._enabled = bool(self._settings.synonym_resolution_enabled)
        self._cache_path = self._settings.synonym_cache_path
        self._http = http_client or httpx.Client(
            timeout=float(self._settings.synonym_request_timeout)
        )
        self._openai = openai_client
        self._lock = threading.RLock()
        self._records: dict[str, SynonymDecision] = {}
        self._wordnet_cache: dict[str, set[str]] = {}
        self._conceptnet_cache: dict[str, set[str]] = {}
        self._embedding_cache: dict[str, list[float]] = {}
        self._last_error = ""
        self._load_cache()

    def discover_equivalences(
        self,
        query_atom: str,
        statements: Iterable[str],
        context: str = "",
    ) -> set[tuple[str, str]]:
        if not self._enabled:
            return set()

        query_expression = parse_expression(query_atom)
        query_terms = self._terms_from_expression(query_expression)
        statement_list = list(statements)
        knowledge_terms: set[str] = set()
        for statement in statement_list:
            expression = parse_expression(statement)
            if (
                isinstance(expression, list)
                and len(expression) >= 3
                and expression[0] == ":"
            ):
                expression = expression[2]
            knowledge_terms.update(self._terms_from_expression(expression))

        query_terms = sorted(query_terms)
        knowledge_terms = sorted(knowledge_terms)
        if not query_terms or not knowledge_terms:
            return set()

        approved: set[tuple[str, str]] = set()
        candidates: dict[tuple[str, str], set[str]] = {}

        for left in query_terms:
            for right in knowledge_terms:
                if equivalent_symbol(left, right):
                    approved.add((left, right))
                    continue
                cached = self._cached_decision(left, right)
                if cached:
                    if cached.relation == "same_meaning":
                        approved.add((left, right))
                    continue
            if any(left == approved_left for approved_left, _ in approved):
                continue
            lexical = self._wordnet_candidates(left)
            lexical.update(self._conceptnet_candidates(left))
            for right in knowledge_terms:
                if canonical_symbol(right) in lexical:
                    candidates.setdefault((left, right), set()).add("lexical")

        remaining_budget = max(
            0,
            int(self._settings.synonym_max_verifications_per_query),
        )
        remaining_budget = self._verify_candidates(
            candidates,
            approved,
            context,
            remaining_budget,
        )

        if remaining_budget > 0 and self._settings.synonym_embedding_enabled:
            embedding_candidates = self._embedding_candidates(
                query_terms,
                knowledge_terms,
                approved,
            )
            remaining_budget = self._verify_candidates(
                embedding_candidates,
                approved,
                context,
                remaining_budget,
            )

        return approved

    def _verify_candidates(
        self,
        candidates: dict[tuple[str, str], set[str]],
        approved: set[tuple[str, str]],
        context: str,
        budget: int,
    ) -> int:
        for pair in sorted(candidates):
            if budget <= 0:
                break
            left, right = pair
            cached = self._cached_decision(left, right)
            if cached:
                if cached.relation == "same_meaning":
                    approved.add(pair)
                continue
            sources = "+".join(sorted(candidates[pair]))
            decision = self._verify_relation(left, right, context, sources)
            budget -= 1
            if decision and decision.relation == "same_meaning":
                approved.add(pair)
        return budget

    def _wordnet_candidates(self, term: str) -> set[str]:
        normalized = canonical_symbol(term)
        if normalized in self._wordnet_cache:
            return set(self._wordnet_cache[normalized])
        candidates: set[str] = set()
        if self._settings.synonym_wordnet_enabled:
            try:
                from nltk.corpus import wordnet

                lookup_forms = {normalized, normalized.replace("_", " ")}
                for lookup in lookup_forms:
                    for synset in wordnet.synsets(lookup):
                        for lemma in synset.lemma_names():
                            candidate = canonical_symbol(lemma)
                            if candidate:
                                candidates.add(candidate)
            except Exception as exc:
                self._set_error(f"WordNet lookup failed: {exc}")
        self._wordnet_cache[normalized] = candidates
        return set(candidates)

    def _conceptnet_candidates(self, term: str) -> set[str]:
        normalized = canonical_symbol(term)
        if normalized in self._conceptnet_cache:
            return set(self._conceptnet_cache[normalized])
        candidates: set[str] = set()
        if self._settings.synonym_conceptnet_lookup_enabled:
            try:
                response = self._http.get(
                    f"{self._settings.synonym_conceptnet_url.rstrip('/')}/query",
                    params={
                        "node": f"/c/en/{normalized}",
                        "rel": "/r/Synonym",
                        "limit": int(self._settings.synonym_conceptnet_limit),
                    },
                )
                response.raise_for_status()
                for edge in response.json().get("edges", []):
                    for endpoint in ("start", "end"):
                        node = edge.get(endpoint, {})
                        if node.get("language") != "en":
                            continue
                        candidate = canonical_symbol(
                            node.get("label")
                            or str(node.get("@id", "")).removeprefix("/c/en/")
                        )
                        if candidate and candidate != normalized:
                            candidates.add(candidate)
            except Exception as exc:
                self._set_error(f"ConceptNet lookup failed: {exc}")
        self._conceptnet_cache[normalized] = candidates
        return set(candidates)

    def _embedding_candidates(
        self,
        query_terms: list[str],
        knowledge_terms: list[str],
        approved: set[tuple[str, str]],
    ) -> dict[tuple[str, str], set[str]]:
        candidates: dict[tuple[str, str], set[str]] = {}
        capped_terms = knowledge_terms[
            : max(0, int(self._settings.synonym_max_knowledge_terms))
        ]
        top_k = max(1, int(self._settings.synonym_embedding_top_k))
        threshold = float(self._settings.synonym_embedding_threshold)

        for left in query_terms:
            if any(left == approved_left for approved_left, _ in approved):
                continue
            left_vector = self._embedding(left)
            if not left_vector:
                continue
            ranked: list[tuple[float, str]] = []
            for right in capped_terms:
                if equivalent_symbol(left, right) or self._cached_decision(left, right):
                    continue
                right_vector = self._embedding(right)
                if not right_vector:
                    continue
                score = self._cosine_similarity(left_vector, right_vector)
                if score >= threshold:
                    ranked.append((score, right))
            for _, right in sorted(ranked, reverse=True)[:top_k]:
                candidates.setdefault((left, right), set()).add("embedding")
        return candidates

    def _embedding(self, term: str) -> list[float]:
        normalized = canonical_symbol(term)
        if normalized in self._embedding_cache:
            return self._embedding_cache[normalized]
        try:
            response = self._http.post(
                self._settings.ollama_url,
                json={
                    "model": self._settings.ollama_model,
                    "prompt": normalized.replace("_", " "),
                },
            )
            response.raise_for_status()
            payload = response.json()
            vector = payload.get("embedding")
            if vector is None:
                embeddings = payload.get("embeddings", [])
                vector = embeddings[0] if embeddings else []
            result = [float(value) for value in vector]
            self._embedding_cache[normalized] = result
            return result
        except Exception as exc:
            self._set_error(f"Ollama embedding failed: {exc}")
            self._embedding_cache[normalized] = []
            return []

    def _verify_relation(
        self,
        left: str,
        right: str,
        context: str,
        sources: str,
    ) -> SynonymDecision | None:
        try:
            client = self._openai or OpenAI(api_key=self._settings.openai_api_key)
            model = str(
                self._settings.synonym_verifier_model
                or self._settings.openai_model
            )
            if model.startswith("openai/"):
                model = model.split("/", 1)[1]
            response = client.responses.create(
                model=model,
                temperature=0,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "Classify the lexical relation between two terms for a logic "
                            "engine. Use same_meaning only when the terms denote the same "
                            "concept and can safely substitute for each other without "
                            "changing truth. Do not call broader, narrower, merely related, "
                            "or contextually associated terms synonyms."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Term A: {left.replace('_', ' ')}\n"
                            f"Term B: {right.replace('_', ' ')}\n"
                            f"Candidate sources: {sources}\n"
                            f"Question context: {context or 'none'}"
                        ),
                    },
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "synonym_relation",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "relation": {
                                    "type": "string",
                                    "enum": sorted(RELATIONS),
                                },
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "reason": {"type": "string"},
                            },
                            "required": ["relation", "confidence", "reason"],
                            "additionalProperties": False,
                        },
                    }
                },
            )
            parsed = json.loads(response.output_text)
            relation = str(parsed["relation"])
            confidence = float(parsed["confidence"])
            if (
                relation == "same_meaning"
                and confidence
                < float(self._settings.synonym_verifier_min_confidence)
            ):
                relation = "ambiguous"
            decision = SynonymDecision(
                left=canonical_symbol(left),
                right=canonical_symbol(right),
                relation=relation,
                confidence=confidence,
                source=f"{sources}+openai",
                reason=str(parsed["reason"]),
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
            self._store_decision(decision)
            self._last_error = ""
            return decision
        except Exception as exc:
            self._set_error(f"OpenAI synonym verification failed: {exc}")
            return None

    def _cached_decision(
        self,
        left: str,
        right: str,
    ) -> SynonymDecision | None:
        with self._lock:
            return self._records.get(self._pair_key(left, right))

    def _store_decision(self, decision: SynonymDecision) -> None:
        key = self._pair_key(decision.left, decision.right)
        with self._lock:
            self._records[key] = decision
            directory = os.path.dirname(self._cache_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            temporary = f"{self._cache_path}.tmp"
            payload = {
                "version": 1,
                "pairs": {
                    item_key: asdict(item)
                    for item_key, item in sorted(self._records.items())
                },
            }
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
            os.replace(temporary, self._cache_path)

    def _load_cache(self) -> None:
        if not os.path.exists(self._cache_path):
            return
        try:
            with open(self._cache_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            records: dict[str, SynonymDecision] = {}
            for key, item in payload.get("pairs", {}).items():
                decision = SynonymDecision(**item)
                if decision.relation in RELATIONS:
                    records[key] = decision
            self._records = records
        except Exception as exc:
            self._set_error(f"Synonym cache load failed: {exc}")

    def _terms_from_expression(self, expression: Any) -> set[str]:
        terms: set[str] = set()

        def visit(value: Any) -> None:
            if not isinstance(value, list) or not value:
                return
            for item in value[1:]:
                if isinstance(item, list):
                    visit(item)
                    continue
                if not isinstance(item, str):
                    continue
                if item.startswith(("$", "?")) or item in IGNORED_SYMBOLS:
                    continue
                normalized = canonical_symbol(item)
                if (
                    normalized
                    and not normalized.replace("_", "").isdigit()
                    and normalized not in {"stv"}
                ):
                    terms.add(normalized)

        visit(expression)
        return terms

    def _pair_key(self, left: str, right: str) -> str:
        normalized = sorted((canonical_symbol(left), canonical_symbol(right)))
        return "|".join(normalized)

    def _cosine_similarity(
        self,
        left: list[float],
        right: list[float],
    ) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if not left_norm or not right_norm:
            return 0.0
        return dot / (left_norm * right_norm)

    def _set_error(self, error: str) -> None:
        self._last_error = error
        print(f"[SynonymResolver] {error}")

    def status(self) -> dict[str, Any]:
        with self._lock:
            same_meaning_count = sum(
                1
                for decision in self._records.values()
                if decision.relation == "same_meaning"
            )
            return {
                "enabled": self._enabled,
                "cached_pairs": len(self._records),
                "cached_synonyms": same_meaning_count,
                "last_error": self._last_error,
            }
