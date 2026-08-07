from __future__ import annotations

import os
import re
from typing import Any, List
from dataclasses import dataclass, field

from config import PROVIDER_PREFIXES, get_settings, normalize_model_id
from core.discourse import ChunkCorefResult, MentionPrepass, MentionPrepassResult
from core.extraction.langextract_chunker import LangExtractChunker
from core.extraction.langextract_examples import load_langextract_prompt_spec
from core.extraction.langextract_pln import (
    build_pln_query,
    collect_predicate_heads,
    format_context_hint,
    log_rejections,
    translate_extractions_to_pln,
    translate_query_extractions_to_pln,
)
from core.pln.postprocessor import PLNPostprocessor
from core.pln.predicate_mapping import (
    LLMPredicateRelationClassifier,
    PredicateMappingEngine,
)
from core.pln.schema_alignment import PLNSchemaAligner
from core.pln.predicate_registry import PredicateRegistry


class ParserProviderError(RuntimeError):
    """
    The LangExtract provider call failed (auth, quota, timeout, bad model id).

    Distinct from a translation error: nothing is wrong with the input, so callers
    should degrade to an explicit no-query result rather than fail the request.
    """


@dataclass
class ParseResult:
    """Result of parsing natural language into PLN."""
    statements: List[str] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class LangExtractPLNParser:
    """
    LangExtract-based parser for PLN-RAG.

    Pipeline:
        natural language -> LangExtract extraction objects -> PLN strings
    """

    def __init__(self):
        cfg = get_settings()
        # The env fallback bypasses Settings, so normalize it here too.
        self._model_id = normalize_model_id(
            _first_nonempty(
                cfg.langextract_model_id,
                os.environ.get("LANGEXTRACT_MODEL_ID"),
                "gemini-2.5-flash",
            )
        )
        self._model_url = _first_nonempty(
            cfg.langextract_model_url,
            os.environ.get("LANGEXTRACT_MODEL_URL"),
        )
        self._api_key = _first_nonempty(
            cfg.langextract_api_key,
            os.environ.get("LANGEXTRACT_API_KEY"),
            cfg.gemini_api_key,
            os.environ.get("GEMINI_API_KEY"),
            cfg.openai_api_key,
            os.environ.get("OPENAI_API_KEY"),
        )
        self._extraction_passes = cfg.langextract_extraction_passes
        self._max_workers = cfg.langextract_max_workers
        self._cache_enabled = cfg.langextract_cache_enabled
        self._cache_max_entries = max(1, cfg.langextract_cache_max_entries)
        self._extraction_cache: dict[tuple[str, str, str], list[Any]] = {}
        self._skip_fuzzy = cfg.langextract_skip_fuzzy
        self._mention_prepass = (
            MentionPrepass() if cfg.mention_prepass_enabled else None
        )
        self._predicate_heads: list[str] = []
        predicate_registry = None
        if cfg.predicate_registry_enabled:
            schema_aligner = PLNSchemaAligner(PLNPostprocessor.STRUCTURAL_HEADS)
            model_id_lower = str(cfg.langextract_model_id or "").lower()
            mapping_gemini_key = cfg.gemini_api_key
            mapping_openai_key = cfg.openai_api_key
            if not mapping_gemini_key and "gemini" in model_id_lower:
                mapping_gemini_key = cfg.langextract_api_key
            if not mapping_openai_key and any(
                marker in model_id_lower for marker in ("openai", "gpt")
            ):
                mapping_openai_key = cfg.langextract_api_key
            classifier = LLMPredicateRelationClassifier(
                enabled=(
                    cfg.predicate_mapping_enabled
                    and cfg.predicate_mapping_llm_enabled
                    and cfg.predicate_mapping_online_enabled
                ),
                gemini_api_key=mapping_gemini_key,
                gemini_model=cfg.gemini_model,
                openai_api_key=mapping_openai_key,
                openai_model=cfg.openai_model,
                timeout_seconds=cfg.predicate_mapping_timeout,
            )
            mapping_engine = (
                PredicateMappingEngine(
                    classifier=classifier,
                    proof_threshold=cfg.predicate_mapping_proof_threshold,
                    retrieval_min_score=cfg.predicate_mapping_min_score,
                    retrieval_top_k=cfg.predicate_mapping_top_k,
                    max_candidates=cfg.predicate_mapping_max_candidates,
                    total_timeout_seconds=cfg.predicate_mapping_total_timeout,
                )
                if cfg.predicate_mapping_enabled
                else None
            )
            predicate_registry = PredicateRegistry(
                path=cfg.predicate_registry_path,
                schema_aligner=schema_aligner,
                mapping_engine=mapping_engine,
            )
        self._postprocessor = PLNPostprocessor(
            predicate_registry=predicate_registry,
            allow_semantic_bridges=cfg.predicate_mapping_emit_bridges,
        )

        if not self._api_key and not self._model_url:
            raise ValueError(
                "No LangExtract model credentials found. Set LANGEXTRACT_API_KEY "
                "or GEMINI_API_KEY, or set LANGEXTRACT_MODEL_URL for a local model."
            )

        self._preflight_model_id()

        spec = load_langextract_prompt_spec(cfg.langextract_examples_path)
        self._statement_prompt = spec.statement_prompt
        self._query_prompt = spec.query_prompt
        self._statement_examples = spec.statement_examples
        self._query_examples = spec.query_examples

    def _preflight_model_id(self) -> None:
        """
        Resolve the model id against LangExtract's provider router at startup.

        Without this, a misconfigured id fails on every extraction call instead,
        which surfaces as empty results or a 500 per request rather than as an
        obvious configuration error.
        """
        if self._model_url:
            # An explicit endpoint bypasses the router entirely.
            return
        try:
            from langextract import providers

            # Providers register lazily; without this the registry is empty and
            # every id looks unresolvable.
            providers.load_builtins_once()
            providers.load_plugins_once()
            router = providers.router
        except Exception as exc:  # pragma: no cover - defensive import
            print(f"[LangExtractPLNParser] provider router unavailable, skipping preflight: {exc}")
            return
        try:
            router.resolve(self._model_id)
        except Exception as exc:
            raise ValueError(
                f"LANGEXTRACT_MODEL_ID={self._model_id!r} is not resolvable by "
                f"LangExtract ({exc}). Provider prefixes are stripped "
                f"automatically for {sorted(PROVIDER_PREFIXES)}, so pass a bare "
                "model id such as 'gemini-2.5-flash' or 'gpt-4o-mini', or set "
                "LANGEXTRACT_MODEL_URL for a local endpoint."
            ) from exc

    def create_chunker(self) -> LangExtractChunker:
        return LangExtractChunker()

    def set_predicate_card_store(self, card_store) -> None:
        self._postprocessor.set_predicate_card_store(card_store)

    def reset(self, clear_registry: bool = False) -> None:
        self._predicate_heads = []
        if clear_registry:
            self._postprocessor.reset_registry()

    def parse(
        self,
        text: str,
        context: list[str],
        mention_prepass: MentionPrepassResult | None = None,
    ) -> ParseResult:
        try:
            mention_prepass = mention_prepass or self._build_mention_prepass(text)
            prompt = self._statement_prompt + format_context_hint(
                context,
                self._predicate_heads,
            ) + self._mention_hint(mention_prepass)
            extractions = self._extract(text, prompt, self._statement_examples)
            self._remember_predicates(collect_predicate_heads(extractions))

            translated = translate_extractions_to_pln(
                extractions,
                source_text=text,
                skip_fuzzy=self._skip_fuzzy,
            )
            processed = self._postprocessor.process(
                text=text,
                statements=translated.statements,
                queries=[],
                context=context,
                plan_queries=False,
            )
            log_rejections("LangExtractPLNParser", translated.rejected)
            return ParseResult(
                statements=processed.statements,
                queries=[],
                metadata={
                    "statement_to_source": _remap_metadata(
                        translated.statements,
                        processed.statements,
                        translated.statement_to_source,
                    ),
                    "rejected": [
                        {
                            "extraction_class": item.extraction_class,
                            "extraction_text": item.extraction_text,
                            "reason": item.reason,
                        }
                        for item in translated.rejected
                    ],
                    "canonicalization_context": translated.ctx,
                    "mention_prepass": mention_prepass.to_dict(),
                    "schema_alignment": processed.alignment_decisions,
                    "predicate_registry": processed.registry_decisions,
                },
            )
        except Exception as exc:
            print(f"[LangExtractPLNParser] Failed for '{text}': {exc}")
            return ParseResult(metadata=_failure_metadata(exc))

    def debug_parse(
        self,
        text: str,
        context: list[str],
        mention_prepass: MentionPrepassResult | None = None,
    ) -> dict[str, Any]:
        mention_prepass = mention_prepass or self._build_mention_prepass(text)
        prompt = self._statement_prompt + format_context_hint(
            context,
            self._predicate_heads,
        ) + self._mention_hint(mention_prepass)
        try:
            extractions = self._extract(text, prompt, self._statement_examples)
        except ParserProviderError as exc:
            print(f"[LangExtractPLNParser] debug_parse provider failure: {exc}")
            return self._empty_debug_statements(mention_prepass, str(exc))
        self._remember_predicates(collect_predicate_heads(extractions))

        translated = translate_extractions_to_pln(
            extractions,
            source_text=text,
            skip_fuzzy=self._skip_fuzzy,
        )
        processed = self._postprocessor.process(
            text=text,
            statements=translated.statements,
            queries=[],
            context=context,
            plan_queries=False,
        )

        return {
            "langextract_postprocessed": {
                "statements": translated.statements,
                "rejected": [
                    {
                        "extraction_class": item.extraction_class,
                        "extraction_text": item.extraction_text,
                        "reason": item.reason,
                    }
                    for item in translated.rejected
                ],
                "canonicalization_context": translated.ctx,
                "mention_prepass": mention_prepass.to_dict(),
                "mention_prompt_hint": self._mention_hint(mention_prepass).strip(),
                "statement_sources": _remap_metadata(
                    translated.statements,
                    processed.statements,
                    translated.statement_to_source,
                ),
            },
            "pln_canonicalized": processed.statements,
            "schema_alignment": processed.alignment_decisions,
            "predicate_registry": processed.registry_decisions,
        }

    def parse_query(self, text: str, context: list[str]) -> ParseResult:
        try:
            mention_prepass = self._build_mention_prepass(text)
            prompt = self._query_prompt + format_context_hint(
                context,
                self._predicate_heads,
            ) + self._mention_hint(mention_prepass)
            extractions = self._extract(text, prompt, self._query_examples)
            translated = translate_query_extractions_to_pln(
                extractions,
                source_text=text,
            )
            processed = self._postprocessor.process(
                text=text,
                statements=translated.statements,
                queries=translated.queries,
                context=context,
                plan_queries=True,
            )
            log_rejections("LangExtractPLNParser.query", translated.rejected)
            return ParseResult(
                statements=processed.statements,
                queries=processed.queries,
                metadata={
                    "query_to_source": _remap_metadata(
                        translated.queries,
                        processed.queries,
                        translated.query_to_source,
                    ),
                    "rejected": [
                        {
                            "extraction_class": item.extraction_class,
                            "extraction_text": item.extraction_text,
                            "reason": item.reason,
                        }
                        for item in translated.rejected
                    ],
                    "canonicalization_context": translated.ctx,
                    "mention_prepass": mention_prepass.to_dict(),
                    "schema_alignment": processed.alignment_decisions,
                    "predicate_registry": processed.registry_decisions,
                },
            )
        except Exception as exc:
            print(f"[LangExtractPLNParser] Query failed for '{text}': {exc}")
            return ParseResult(metadata=_failure_metadata(exc))

    def debug_parse_query(self, text: str, context: list[str]) -> dict[str, Any]:
        mention_prepass = self._build_mention_prepass(text)
        prompt = self._query_prompt + format_context_hint(
            context,
            self._predicate_heads,
        ) + self._mention_hint(mention_prepass)
        try:
            extractions = self._extract(text, prompt, self._query_examples)
        except ParserProviderError as exc:
            print(f"[LangExtractPLNParser] debug_parse_query provider failure: {exc}")
            return self._empty_debug_queries(mention_prepass, str(exc))
        translated = translate_query_extractions_to_pln(
            extractions,
            source_text=text,
        )
        processed = self._postprocessor.process(
            text=text,
            statements=translated.statements,
            queries=translated.queries,
            context=context,
            plan_queries=True,
        )

        return {
            "langextract_postprocessed": {
                "queries": translated.queries,
                "rejected": [
                    {
                        "extraction_class": item.extraction_class,
                        "extraction_text": item.extraction_text,
                        "reason": item.reason,
                    }
                    for item in translated.rejected
                ],
                "canonicalization_context": translated.ctx,
                "mention_prepass": mention_prepass.to_dict(),
                "mention_prompt_hint": self._mention_hint(mention_prepass).strip(),
                "query_sources": translated.query_to_source,
            },
            "pln_canonicalized": processed.queries,
            "supporting_statements": processed.statements,
            "schema_alignment": processed.alignment_decisions,
            "predicate_registry": processed.registry_decisions,
        }

    def _empty_debug_statements(
        self,
        mention_prepass: MentionPrepassResult,
        provider_error: str,
    ) -> dict[str, Any]:
        return {
            "langextract_postprocessed": {
                "statements": [],
                "rejected": [],
                "canonicalization_context": {},
                "mention_prepass": mention_prepass.to_dict(),
                "mention_prompt_hint": self._mention_hint(mention_prepass).strip(),
                "statement_sources": {},
                "provider_error": provider_error,
            },
            "pln_canonicalized": [],
            "schema_alignment": [],
            "predicate_registry": [],
        }

    def _empty_debug_queries(
        self,
        mention_prepass: MentionPrepassResult,
        provider_error: str,
    ) -> dict[str, Any]:
        return {
            "langextract_postprocessed": {
                "queries": [],
                "rejected": [],
                "canonicalization_context": {},
                "mention_prepass": mention_prepass.to_dict(),
                "mention_prompt_hint": self._mention_hint(mention_prepass).strip(),
                "query_sources": {},
                "provider_error": provider_error,
            },
            "pln_canonicalized": [],
            "supporting_statements": [],
            "schema_alignment": [],
            "predicate_registry": [],
        }

    def _extract(self, text: str, prompt: str, examples: list[Any]) -> list[Any]:
        import langextract as lx

        cache_key = (self._model_id, text, prompt)
        cached = self._extraction_cache.get(cache_key) if self._cache_enabled else None
        if cached is not None:
            return list(cached)

        try:
            result = lx.extract(
                text_or_documents=text,
                prompt_description=prompt,
                examples=examples,
                model_id=self._model_id,
                model_url=self._model_url,
                api_key=self._api_key,
                extraction_passes=self._extraction_passes,
                max_char_buffer=max(len(text) + 1, 1000),
                max_workers=self._max_workers,
                show_progress=False,
            )
        except Exception as exc:
            # Single choke point for provider failures so every caller degrades
            # the same way instead of some raising and some returning empty.
            raise ParserProviderError(
                f"{type(exc).__name__}: {exc}"
            ) from exc
        extracted = list(result.extractions) if hasattr(result, "extractions") else []
        if self._cache_enabled:
            if len(self._extraction_cache) >= self._cache_max_entries:
                self._extraction_cache.pop(next(iter(self._extraction_cache)))
            self._extraction_cache[cache_key] = list(extracted)
        return extracted

    def _remember_predicates(self, heads: list[str]) -> None:
        for head in heads:
            if head not in self._predicate_heads:
                self._predicate_heads.append(head)

    def build_mention_prepass(
        self,
        text: str,
        *,
        coref_result: ChunkCorefResult | None = None,
        global_offset: int = 0,
    ) -> MentionPrepassResult:
        return self._build_mention_prepass(
            text,
            coref_result=coref_result,
            global_offset=global_offset,
        )

    def _build_mention_prepass(
        self,
        text: str,
        *,
        coref_result: ChunkCorefResult | None = None,
        global_offset: int = 0,
    ) -> MentionPrepassResult:
        if not self._mention_prepass:
            return MentionPrepassResult()
        return self._mention_prepass.build(
            text,
            coref_result=coref_result,
            global_offset=global_offset,
        )

    def _mention_hint(self, result: MentionPrepassResult) -> str:
        return result.prompt_hint()


def _failure_metadata(exc: Exception) -> dict[str, Any]:
    """Carry the failure reason out of parse/parse_query instead of losing it."""
    message = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, ParserProviderError):
        return {"provider_error": str(exc)}
    return {"parse_error": message}


def _first_nonempty(*values: Any) -> str | None:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _remap_metadata(
    before: list[str],
    after: list[str],
    metadata: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    remapped: dict[str, dict[str, Any]] = {}
    unmatched_before: set[str] = set(before)
    unmatched_after: list[str] = []
    for item in after:
        if item in metadata:
            remapped[item] = {
                **metadata[item],
                "lineage_relation": "extracted_from",
                "lineage_score": 1.0,
            }
            unmatched_before.discard(item)
            continue
        unmatched_after.append(item)

    # Normalization can rename or reformat an atom. Preserve provenance only
    # when its structural token set has one strong, unique source match. This
    # avoids the previous positional mapping, which could assign a later source
    # to an inferred or reordered atom.
    candidates: list[tuple[float, str, str]] = []
    for item in unmatched_after:
        after_tokens = _lineage_tokens(item)
        if not after_tokens:
            continue
        for source_item in unmatched_before:
            if source_item not in metadata:
                continue
            before_tokens = _lineage_tokens(source_item)
            union = before_tokens | after_tokens
            score = len(before_tokens & after_tokens) / len(union) if union else 0.0
            if score >= 0.72:
                candidates.append((score, item, source_item))

    used_after: set[str] = set()
    used_before: set[str] = set()
    for score, item, source_item in sorted(candidates, reverse=True):
        if item in used_after or source_item in used_before:
            continue
        remapped[item] = {
            **metadata[source_item],
            "lineage_relation": "normalized_from",
            "lineage_score": round(score, 4),
            "source_atom": source_item,
        }
        used_after.add(item)
        used_before.add(source_item)
    return remapped


def _lineage_tokens(statement: str) -> set[str]:
    clean = " ".join(str(statement).split())
    clean = re.sub(r"^\(:\s+[^\s()]+\s+", "", clean)
    clean = re.sub(r"\(STV\s+[^()]+\)\s*\)?$", "", clean)
    ignored = {
        "implication",
        "premises",
        "conclusions",
        "stv",
        "not",
        "and",
        "or",
    }
    return {
        token.lower()
        for token in re.findall(r"[$?]?[A-Za-z][A-Za-z0-9_]*|[0-9]+", clean)
        if token.lower() not in ignored
    }


__all__ = [
    "LangExtractPLNParser",
    "build_pln_query",
    "translate_extractions_to_pln",
    "translate_query_extractions_to_pln",
]
