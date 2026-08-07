import asyncio
import re
import time
from dataclasses import replace
from typing import List

from config import get_settings
from parsers.langextract_pln_parser import LangExtractPLNParser
from core.query.alignment import (
    build_aligned_queries,
    extract_forward_seed_terms,
    extract_query_targets,
    filter_queries_by_question_intent_verbose,
)
from core.query.intent import (
    QuestionIntent,
    QuestionMode,
    evaluate_query_intent,
    parse_question_intent,
    parse_query_signature,
)
from core.query.target_gate import SemanticTargetGate
from core.reasoning.reasoner import ProofOutcome, Reasoner
from core.pln.symbol_normalization import canonical_symbol
from core.answering.answer_generator import AnswerGenerator
from core.discourse import (
    CoreferenceResolver,
    DocumentCorefResult,
    NullCoreferenceResolver,
    project_coref_to_chunk,
)
from core.extraction.langextract_chunker import TextChunk
from core.evidence import ClaimEvidenceRecord, build_claim_evidence_record
from storage.evidence_ledger import EvidenceLedger
from storage.vector_store import VectorStore
from api.models import (
    IngestItemResult,
    QueryResponse,
    DebugIngestItemResult,
    DebugIngestChunkResult,
    LangExtractPostprocessed,
    LangExtractQueryPostprocessed,
    DebugQueryResponse,
)


class PLNRAGService:
    """
    Orchestrates the full LangExtract pipeline:
      Text → Chunker → LangExtractPLNParser → Reasoner → AnswerGenerator
    """

    def __init__(self, parser: LangExtractPLNParser):
        cfg = get_settings()
        self._parser = parser
        self._chunker = parser.create_chunker()
        self._reasoner = Reasoner()
        self._vector_store = VectorStore() if cfg.use_vector_store else None
        if self._vector_store and hasattr(parser, "set_predicate_card_store"):
            parser.set_predicate_card_store(self._vector_store)
        self._answer_gen = AnswerGenerator()
        self._context_top_k = cfg.context_top_k
        self._query_fallback_enabled = cfg.query_fallback_enabled
        self._coreference_enabled = cfg.coreference_enabled
        self._coreference_fail_open = cfg.coreference_fail_open
        self._coreference_resolver = self._create_coreference_resolver(cfg)
        self._evidence_ledger = (
            EvidenceLedger(cfg.evidence_ledger_path)
            if cfg.evidence_ledger_enabled
            else None
        )
        self._evidence_outbox_batch_size = cfg.evidence_outbox_batch_size
        self._proof_equivalent_only = cfg.query_proof_equivalent_only
        self._target_gate = self._create_target_gate(cfg)
        # Targets the gate judged to be the negation of what was asked; the
        # executor runs the negated form for these.
        self._negation_targets: set[str] = set()
        # Enumerated answers for the most recent open-question target.
        self._last_bindings = None
        if self._evidence_ledger and self._evidence_ledger.claim_count:
            self._rebuild_reasoner_from_ledger()
        self._flush_evidence_index()

    def _create_target_gate(self, cfg) -> SemanticTargetGate | None:
        """
        Build the semantic gate that decides which targets the question asks about.

        Embeddings reuse the vector store already talking to Ollama. The predicate
        card lookup is optional: when a card exists its label and definition make a
        better gloss than the predicate name alone.
        """
        if not getattr(cfg, "query_target_gate_enabled", True):
            print("[TargetGate] disabled; every structural candidate is admitted.")
            return None
        gate = SemanticTargetGate(
            enabled=True,
            embedder=self._vector_store,
            card_lookup=self._predicate_card_lookup,
            gemini_api_key=cfg.gemini_api_key or cfg.langextract_api_key,
            gemini_model=cfg.gemini_model,
            openai_api_key=cfg.openai_api_key,
            openai_model=cfg.openai_model,
            top_k=cfg.query_target_gate_top_k,
            timeout_seconds=cfg.query_target_gate_timeout,
            min_confidence=cfg.query_target_gate_min_confidence,
            negation_min_confidence=cfg.query_target_gate_negation_min_confidence,
        )
        print(
            "[TargetGate] enabled "
            f"top_k={gate.top_k} min_confidence={gate.min_confidence} "
            f"llm={'yes' if gate.llm_available else 'no (embedding rank only)'}"
        )
        return gate

    def _predicate_card_lookup(self, predicate: str, arity: int) -> dict | None:
        registry = getattr(
            getattr(self._parser, "_postprocessor", None),
            "_predicate_registry",
            None,
        )
        if not registry or not hasattr(registry, "card_for"):
            return None
        try:
            card = registry.card_for(predicate, arity)
        except Exception:
            return None
        if not card:
            return None
        return {
            "label": (card.labels[0] if card.labels else ""),
            "definition": card.definition,
            "argument_types": list(card.argument_types),
        }

    def _create_coreference_resolver(self, cfg) -> CoreferenceResolver:
        if not cfg.coreference_enabled:
            print("[Coreference] disabled.")
            return NullCoreferenceResolver()
        if cfg.coreference_backend != "lingmess":
            message = f"Unsupported coreference backend: {cfg.coreference_backend}"
            if cfg.coreference_fail_open:
                print(f"[Coreference] {message}; using deterministic prepass.")
                return NullCoreferenceResolver()
            raise ValueError(message)
        try:
            from core.discourse.lingmess_coreference import (
                LingMessCoreferenceResolver,
            )

            print(
                "[Coreference] enabled backend=lingmess "
                f"model={cfg.coreference_model} mode={cfg.coreference_mode} "
                f"device={cfg.coreference_device}"
            )
            return LingMessCoreferenceResolver(
                model_name=cfg.coreference_model,
                device=cfg.coreference_device,
                max_tokens_in_batch=cfg.coreference_max_tokens_in_batch,
                include_pair_logits=cfg.coreference_include_pair_logits,
            )
        except Exception as exc:
            if cfg.coreference_fail_open:
                print(
                    "[Coreference] initialization failed; using deterministic "
                    f"prepass: {exc}"
                )
                return NullCoreferenceResolver()
            raise

    def _record_chunk_claims(
        self,
        *,
        document_id: str,
        chunk: TextChunk,
        statements: List[str],
        metadata: dict,
    ) -> List[ClaimEvidenceRecord]:
        ledger = getattr(self, "_evidence_ledger", None)
        if not ledger:
            return []
        mention_prepass = metadata.get("mention_prepass", {})
        evidence_id = ledger.record_evidence(
            document_id=document_id,
            chunk_index=chunk.index,
            start_pos=chunk.start,
            end_pos=chunk.end,
            text=chunk.text,
            coreference=mention_prepass.get("coreference", {}),
        )
        source_map = metadata.get("statement_to_source", {})
        records = [
            build_claim_evidence_record(
                document_id=document_id,
                evidence_id=evidence_id,
                atom=statement,
                chunk_text=chunk.text,
                chunk_index=chunk.index,
                chunk_start=chunk.start,
                source=source_map.get(statement),
                mention_prepass=mention_prepass,
            )
            for statement in statements
        ]
        ledger.record_claims(records)
        return records

    def _reasoner_inputs(
        self,
        statements: List[str],
        records: List[ClaimEvidenceRecord],
        fallback_provenance: dict,
    ) -> tuple[List[str], dict[str, dict]]:
        if not getattr(self, "_evidence_ledger", None):
            return statements, fallback_provenance
        eligible = [record for record in records if record.is_reasoner_eligible]
        return (
            [record.atom for record in eligible],
            {record.atom: record.source_metadata() for record in eligible},
        )

    def _flush_evidence_index(self) -> int:
        ledger = getattr(self, "_evidence_ledger", None)
        vector_store = getattr(self, "_vector_store", None)
        if not ledger or not vector_store:
            return 0
        records = ledger.pending_index_records(
            limit=getattr(self, "_evidence_outbox_batch_size", 256),
        )
        if not records:
            return 0
        outbox_ids = [int(record["outbox_id"]) for record in records]
        try:
            vector_store.store_claim_records(records)
            ledger.mark_indexed(outbox_ids)
            return len(records)
        except Exception as exc:
            ledger.mark_index_failed(outbox_ids, str(exc))
            print(f"[EvidenceIndex] deferred {len(records)} records: {exc}")
            return 0

    def _retrieve_ingest_context(self, text: str) -> tuple[List[str], List[float]]:
        vector_store = getattr(self, "_vector_store", None)
        if not vector_store:
            return [], []
        try:
            return vector_store.retrieve_context(text, top_k=self._context_top_k)
        except Exception as exc:
            print(f"[VectorStore] ingest context unavailable; continuing: {exc}")
            return [], []

    def _rebuild_reasoner_from_ledger(self) -> int:
        ledger = getattr(self, "_evidence_ledger", None)
        if not ledger:
            return 0
        rows = ledger.reasoner_records()
        self._reasoner.reset()
        if not rows:
            return 0
        statements = [row["atom"] for row in rows]
        provenance = {row["atom"]: row["source"] for row in rows}
        return len(self._reasoner.add_statements(statements, provenance=provenance))

    #  Ingest

    async def ingest_batch(self, texts: List[str]) -> List[IngestItemResult]:
        """
        Process texts sequentially so each sentence can see
        all previously ingested atoms as context.
        """
        results = []
        for text in texts:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._ingest_single, text
            )
            results.append(result)
        return results

    async def debug_ingest_batch(self, texts: List[str]) -> List[DebugIngestItemResult]:
        results = []
        for text in texts:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._debug_ingest_single, text
            )
            results.append(result)
        return results

    def _ingest_single(self, text: str) -> IngestItemResult:
        try:
            # 1. Chunk large texts
            chunks = self._chunk_document(text)
            document_coref = self._resolve_document_coref(text)
            ledger = getattr(self, "_evidence_ledger", None)
            document_id = ledger.record_document(text) if ledger else ""
            all_atoms: List[str] = []
            parser_calls = 0
            rejected_samples: List[str] = []

            for chunk in chunks:
                chunk_text = chunk.text
                mention_prepass = self._build_chunk_mention_prepass(
                    chunk,
                    document_coref,
                )
                # 2. Retrieve context from atomspace + vector store
                context, vector = self._retrieve_ingest_context(chunk_text)
                # Also supplement with recent atoms from disk
                context = self._enrich_context(context)

                # 3. Parse chunk → PLN atoms
                parse_result = self._parse_chunk(chunk_text, context, mention_prepass)
                parser_calls += 1

                if not parse_result.statements:
                    print(f"[Service] No statements for chunk: '{chunk_text[:60]}...'")
                    continue

                # 4. Validate and record claim/evidence lineage before indexing.
                records = self._record_chunk_claims(
                    document_id=document_id,
                    chunk=chunk,
                    statements=parse_result.statements,
                    metadata=parse_result.metadata,
                )
                reasoner_statements, provenance = self._reasoner_inputs(
                    parse_result.statements,
                    records,
                    parse_result.metadata.get("statement_to_source", {}),
                )
                rejected_samples.extend(
                    f"{record.atom}: {'; '.join(record.validation_errors)}"
                    for record in records
                    if record.validation_state == "quarantined"
                )

                # 5. Add only accepted or traceable derived claims to Atomspace.
                added = self._reasoner.add_statements(
                    reasoner_statements,
                    provenance=provenance,
                )

                all_atoms.extend(added)

                # 6. Legacy path only. Evidence-ledger records are indexed from
                # the transactional outbox and contain one target per point.
                if added and self._vector_store and not ledger:
                    self._vector_store.store(
                        chunk_text,
                        added,
                        vector,
                        metadata=parse_result.metadata,
                        query_targets=extract_query_targets(added),
                    )
                self._flush_evidence_index()

            return IngestItemResult(
                text=text,
                atoms=all_atoms,
                status="success",
                chunk_count=len(chunks),
                parser_calls=parser_calls,
                rejected_count=len(rejected_samples),
                rejected_samples=rejected_samples[:10],
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            return IngestItemResult(text=text, status="failed", error=str(e))

    def _debug_ingest_single(self, text: str) -> DebugIngestItemResult:
        try:
            chunks = self._chunk_document(text)
            document_coref = self._resolve_document_coref(text)
            ledger = getattr(self, "_evidence_ledger", None)
            document_id = ledger.record_document(text) if ledger else ""
            chunk_results: List[DebugIngestChunkResult] = []

            for chunk in chunks:
                chunk_text = chunk.text
                mention_prepass = self._build_chunk_mention_prepass(
                    chunk,
                    document_coref,
                )
                context, vector = self._retrieve_ingest_context(chunk_text)
                context = self._enrich_context(context)

                debug_info = self._debug_parse_chunk(
                    chunk_text,
                    context,
                    mention_prepass,
                )
                langextract_info = debug_info.get("langextract_postprocessed", {})
                pln_canonicalized = debug_info.get("pln_canonicalized", [])
                schema_alignment = debug_info.get("schema_alignment", [])
                predicate_registry = debug_info.get("predicate_registry", [])
                debug_metadata = {
                    "statement_to_source": langextract_info.get(
                        "statement_sources",
                        {},
                    ),
                    "rejected": langextract_info.get("rejected", []),
                    "canonicalization_context": langextract_info.get(
                        "canonicalization_context",
                        {},
                    ),
                    "mention_prepass": langextract_info.get("mention_prepass", {}),
                    "schema_alignment": schema_alignment,
                    "predicate_registry": predicate_registry,
                }

                records = self._record_chunk_claims(
                    document_id=document_id,
                    chunk=chunk,
                    statements=pln_canonicalized,
                    metadata=debug_metadata,
                )
                reasoner_statements, provenance = self._reasoner_inputs(
                    pln_canonicalized,
                    records,
                    debug_metadata.get("statement_to_source", {}),
                )
                added = self._reasoner.add_statements(
                    reasoner_statements,
                    provenance=provenance,
                )

                if added and self._vector_store and not ledger:
                    self._vector_store.store(
                        chunk_text,
                        added,
                        vector,
                        metadata=debug_metadata,
                        query_targets=extract_query_targets(added),
                    )
                self._flush_evidence_index()

                chunk_results.append(
                    DebugIngestChunkResult(
                        chunk=chunk_text,
                        chunk_start=chunk.start,
                        chunk_end=chunk.end,
                        coreference=getattr(mention_prepass, "coreference", {}),
                        context=context,
                        langextract_postprocessed=LangExtractPostprocessed(
                            **langextract_info
                        ),
                        pln_canonicalized=pln_canonicalized,
                        atomspace_added=added,
                        evidence_records=[record.ledger_dict() for record in records],
                        schema_alignment=schema_alignment,
                        predicate_registry=predicate_registry,
                    )
                )

            return DebugIngestItemResult(
                text=text,
                chunks=chunk_results,
                coreference=document_coref.to_dict(),
                status="success",
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            return DebugIngestItemResult(text=text, status="failed", error=str(e))

    def _chunk_document(self, text: str) -> List[TextChunk]:
        if hasattr(self._chunker, "chunk_with_spans"):
            return list(self._chunker.chunk_with_spans(text))
        chunks = self._chunker.chunk(text)
        result: List[TextChunk] = []
        cursor = 0
        for index, chunk in enumerate(chunks):
            start = text.find(chunk, cursor)
            if start < 0:
                start = cursor
            end = min(len(text), start + len(chunk))
            cursor = end
            result.append(TextChunk(index=index, text=chunk, start=start, end=end))
        return result

    def _resolve_document_coref(self, text: str) -> DocumentCorefResult:
        if not self._coreference_enabled:
            return DocumentCorefResult(text=text, backend="none", model_name="")
        try:
            return self._coreference_resolver.resolve_document(text)
        except Exception as exc:
            if not self._coreference_fail_open:
                raise
            print(
                "[Coreference] document resolution failed; continuing with "
                f"deterministic prepass: {exc}"
            )
            return DocumentCorefResult(
                text=text,
                backend=getattr(self._coreference_resolver, "backend", "unknown"),
                model_name=getattr(self._coreference_resolver, "model_name", ""),
                error=str(exc),
            )

    def _build_chunk_mention_prepass(
        self,
        chunk: TextChunk,
        document_coref: DocumentCorefResult,
    ):
        chunk_coref = project_coref_to_chunk(
            document_coref,
            chunk.start,
            chunk.end,
            chunk.text,
        )
        if hasattr(self._parser, "build_mention_prepass"):
            return self._parser.build_mention_prepass(
                chunk.text,
                coref_result=chunk_coref,
                global_offset=chunk.start,
            )
        return None

    def _parse_chunk(self, text: str, context: List[str], mention_prepass):
        try:
            return self._parser.parse(
                text,
                context,
                mention_prepass=mention_prepass,
            )
        except TypeError:
            return self._parser.parse(text, context)

    def _debug_parse_chunk(self, text: str, context: List[str], mention_prepass):
        try:
            return self._parser.debug_parse(
                text,
                context,
                mention_prepass=mention_prepass,
            )
        except TypeError:
            return self._parser.debug_parse(text, context)

    def _enrich_context(self, rag_context: List[str], max_atoms: int = 50) -> List[str]:
        """
        Supplement RAG-retrieved context with the most recent atoms
        from the atomspace file, deduplicating and capping at max_atoms.
        This ensures the parser always has the full predicate vocabulary
        even when RAG similarity scores are low.
        """
        from config import get_settings
        import os

        cfg = get_settings()
        file_atoms: List[str] = []
        if os.path.exists(cfg.atomspace_path):
            with open(cfg.atomspace_path, "r") as f:
                file_atoms = [l.strip() for l in f if l.strip()]

        seen = set()
        merged = []
        for atom in file_atoms + rag_context:
            if atom not in seen:
                seen.add(atom)
                merged.append(atom)

        return merged[-max_atoms:]

    #  Query

    def _qdrant_context_with_matches(
        self,
        text: str,
        top_k: int,
    ) -> tuple[List[str], List[dict]]:
        if not self._vector_store:
            return [], []
        try:
            matches, _ = self._vector_store.search(text, top_k=top_k)
        except Exception as exc:
            print(f"[VectorStore] query context unavailable; continuing: {exc}")
            return [], []
        context: List[str] = []
        for item in matches:
            pln = item.get("pln", [])
            if isinstance(pln, list):
                context.extend(pln)
        return context, matches

    def _alignment_matches(self, matches: List[dict]) -> List[dict]:
        cfg = get_settings()
        min_score = float(getattr(cfg, "query_alignment_min_score", 0.0) or 0.0)
        hybrid_min_score = float(
            getattr(cfg, "query_alignment_hybrid_min_score", min_score) or 0.0
        )
        return [
            match
            for match in matches
            if float(match.get("score") or 0.0) >= (
                hybrid_min_score
                if match.get("score_kind") == "hybrid_rrf"
                else min_score
            )
            and (
                not getattr(self, "_evidence_ledger", None)
                or (
                    match.get("validation_state") == "accepted"
                    and bool(match.get("claim_evidence_id"))
                )
            )
        ]

    def _query_candidates(
        self,
        question: str,
        qdrant_queries: List[str],
        parser_queries: List[str],
        trusted_queries: List[str] | None = None,
    ) -> tuple[List[tuple[str, str]], List[dict]]:
        """
        Rank executable query candidates, returning (entries, rejections).

        Structural admissibility is checked first (well-formed target, question
        mode, KB arity), then the semantic gate decides which surviving targets the
        question actually asks about and in what order. Every discarded candidate
        is recorded with the gate that discarded it, so an empty result can be
        explained instead of surfacing as an opaque
        `no_semantically_compatible_query`.
        """
        entries: List[tuple[str, str]] = []
        rejections: List[dict] = []
        seen: set[str] = set()
        intent = parse_question_intent(question)
        filtered_parser, parser_rejected = filter_queries_by_question_intent_verbose(
            question,
            parser_queries,
        )
        filtered_qdrant, qdrant_rejected = filter_queries_by_question_intent_verbose(
            question,
            qdrant_queries,
        )
        for source, dropped in (
            ("parser", parser_rejected),
            ("qdrant_alignment", qdrant_rejected),
        ):
            rejections.extend(
                {"query": query, "source": source, "stage": stage, "detail": detail}
                for query, stage, detail in dropped
            )

        # Qdrant candidates originate from accepted claim-evidence records.
        # They still pass the same intent and KB arity gates as parser output.
        for source, queries in (
            ("deterministic", trusted_queries or []),
            ("qdrant_alignment", filtered_qdrant),
            ("parser", filtered_parser),
        ):
            for query in queries:
                clean = " ".join(str(query).split())
                if not clean or clean in seen:
                    continue
                verdict = evaluate_query_intent(intent, clean)
                if not verdict.accepted:
                    rejections.append({
                        "query": clean,
                        "source": source,
                        "stage": verdict.stage,
                        "detail": verdict.detail,
                    })
                    continue
                allowed, arity_stage, arity_detail = self._query_arity_verdict(clean)
                if not allowed:
                    rejections.append({
                        "query": clean,
                        "source": source,
                        "stage": arity_stage,
                        "detail": arity_detail,
                    })
                    continue
                seen.add(clean)
                entries.append((clean, source))

        return self._apply_semantic_gate(question, entries, rejections)

    def _apply_semantic_gate(
        self,
        question: str,
        entries: List[tuple[str, str]],
        rejections: List[dict],
    ) -> tuple[List[tuple[str, str]], List[dict]]:
        """
        Order and filter structurally valid candidates by what the question means.

        Ordering comes from the gate, which ranks by embedding similarity before
        asking the model for a verdict. Candidates the gate marks `asks_negation`
        are kept and remembered, so the executor can run the negated form.
        """
        gate = getattr(self, "_target_gate", None)
        if not gate or not entries:
            self._negation_targets = set()
            return entries, rejections

        source_by_query = {query: source for query, source in entries}
        # `deterministic` candidates are compiled from atoms the reasoner actually
        # holds, so they are the only ones that can be proved. Tell the gate.
        kb_grounded = [
            query for query, source in entries if source == "deterministic"
        ]
        decision = gate.decide(
            question,
            [query for query, _source in entries],
            kb_grounded=kb_grounded,
        )

        for query, stage, detail in decision.rejections:
            rejections.append({
                "query": query,
                "source": source_by_query.get(query, "unknown"),
                "stage": stage,
                "detail": detail,
            })
        if not decision.gate_available:
            rejections.append({
                "query": "",
                "source": "semantic_gate",
                "stage": "gate_unavailable",
                "detail": (
                    "semantic target gate did not return verdicts; candidates kept "
                    "on embedding rank with no veto"
                ),
            })

        self._negation_targets = {
            target
            for target, verdict in decision.verdicts.items()
            if verdict.wants_negative_execution
        }
        return (
            [(query, source_by_query.get(query, "unknown")) for query in decision.admitted],
            rejections,
        )

    def _deterministic_query_candidates(
        self,
        question: str,
        allow_open_targets: bool = False,
    ) -> List[str]:
        """
        Compile candidate targets from atoms the knowledge base actually holds.

        With `allow_open_targets`, signatures that cannot be fully grounded from the
        question are also emitted with their variable intact. "Which biomarkers
        predict severity" has no entity to substitute, so a fully grounded target
        cannot express it — the open form does, and the reasoner enumerates values.
        """
        reasoner = getattr(self, "_reasoner", None)
        if not reasoner or not hasattr(reasoner, "proposition_signatures"):
            return []
        signatures = reasoner.proposition_signatures()
        entities = self._mentioned_kb_entities(question, signatures)
        candidates: List[str] = []
        seen: set[str] = set()
        for signature in signatures:
            args = list(signature.get("args", []))
            variables = list(signature.get("variables", []))
            grounded_rows: List[List[str]] = []
            if variables:
                if len(set(variables)) != 1:
                    continue
                grounded_rows = [
                    [entity if arg in variables else arg for arg in args]
                    for entity in entities
                ]
            else:
                grounded_rows = [args]
            if allow_open_targets and variables and len(set(variables)) == 1:
                # Keep the variable so the reasoner can enumerate answers.
                grounded_rows = grounded_rows + [args]
            for grounded in grounded_rows:
                has_variable = any(arg.startswith(("$", "?")) for arg in grounded)
                if has_variable and not allow_open_targets:
                    continue
                query = (
                    f"(: $prf ({signature['head']} {' '.join(grounded)}) $tv)"
                )
                if query in seen:
                    continue
                seen.add(query)
                candidates.append(query)
        return candidates

    def _mentioned_kb_entities(
        self,
        question: str,
        signatures: List[dict],
    ) -> List[str]:
        normalized_question = canonical_symbol(question, lemmatize=False)
        compact_question = normalized_question.replace("_", "")
        entities: List[str] = []
        for signature in signatures:
            for raw_arg in signature.get("args", []):
                arg = canonical_symbol(str(raw_arg), lemmatize=False)
                if not arg or arg.startswith(("$", "?")) or arg.isdigit():
                    continue
                phrase_match = re.search(
                    rf"(?:^|_){re.escape(arg)}(?:_|$)",
                    normalized_question,
                )
                compact_match = len(arg) >= 4 and arg.replace("_", "") in compact_question
                if (phrase_match or compact_match) and arg not in entities:
                    entities.append(arg)
        return sorted(entities, key=lambda item: (-len(item), item))

    def _query_arity_allowed(self, query: str) -> bool:
        return self._query_arity_verdict(query)[0]

    def _query_arity_verdict(self, query: str) -> tuple[bool, str, str]:
        """Check the target head/arity against the KB, reporting why on failure."""
        signature = parse_query_signature(query)
        if not signature:
            return False, "malformed", "query is not a (: $prf (Head args) $tv) form"
        reasoner = getattr(self, "_reasoner", None)
        if not reasoner or not hasattr(reasoner, "predicate_arities"):
            return True, "", ""
        arities = reasoner.predicate_arities()
        head = str(signature["head"])
        allowed = arities.get(head)
        if arities and allowed is None:
            # An unobserved head used to be rejected outright. It cannot produce a
            # wrong answer — the proof simply fails — so admit it and let the
            # semantic gate rank it. Rejecting here made any question whose
            # predicate the KB had not seen unanswerable by construction.
            return True, "", ""
        if not allowed or int(signature["arity"]) in allowed:
            return True, "", ""
        return (
            False,
            "arity_conflict",
            f"{head} was observed with arity {sorted(allowed)}, "
            f"target uses {signature['arity']}",
        )

    async def query(self, question: str) -> QueryResponse:
        cfg = get_settings()
        intent = parse_question_intent(question)
        # 1. Retrieve context for translation
        t0 = time.perf_counter()
        if self._vector_store:
            context, qdrant_matches = self._qdrant_context_with_matches(
                question,
                top_k=max(
                    self._context_top_k,
                    int(getattr(cfg, "query_alignment_top_k", 0) or 0),
                ),
            )
        else:
            context, qdrant_matches = [], []
        alignment_matches = self._alignment_matches(qdrant_matches)
        qdrant_alignment = (
            build_aligned_queries(question, alignment_matches)
            if getattr(cfg, "query_alignment_enabled", True)
            else None
        )
        qdrant_queries = qdrant_alignment.queries if qdrant_alignment else []
        seed_terms = extract_forward_seed_terms(alignment_matches)
        context = self._enrich_context(context)
        context_retrieval_seconds = time.perf_counter() - t0

        # 2. Parse question → PLN query
        t1 = time.perf_counter()
        parse_result = self._parser.parse_query(question, context)
        parse_query_seconds = time.perf_counter() - t1

        original_query = parse_result.queries[0] if parse_result.queries else (
            qdrant_queries[0] if qdrant_queries else ""
        )
        trusted_queries = self._deterministic_query_candidates(
            question,
            allow_open_targets=intent.mode == QuestionMode.OPEN,
        )
        candidate_entries, candidate_rejections = self._query_candidates(
            question,
            qdrant_queries,
            parse_result.queries,
            trusted_queries,
        )
        if intent.mode in {QuestionMode.FACTORS, QuestionMode.SUFFICIENCY}:
            requirements = self._reasoner.explain_requirements(
                set(intent.terms),
                entity=intent.entities[0] if intent.entities else "",
                direction=intent.direction,
            )
            answer = self._answer_structured_intent(intent, requirements)
            return QueryResponse(
                question=question,
                pln_query="",
                original_query=original_query,
                executed_query="",
                query_source="none",
                fallback_used=False,
                query_status="well_aligned" if requirements else "weakly_aligned",
                raw_proof="",
                sources=[],
                answer=answer,
                qdrant_aligned_queries=qdrant_queries,
                execution_candidates=[],
                intent_mode=intent.mode.value,
                proof_status="unanswered",
                proof_validated=True,
                requirements=requirements,
                context_retrieval_seconds=round(context_retrieval_seconds, 4),
                parse_query_seconds=round(parse_query_seconds, 4),
                reasoning_seconds=0.0,
                source_lookup_seconds=0.0,
                answer_generation_seconds=0.0,
            )
        if not candidate_entries:
            # Report what the parser and alignment actually produced. Reporting
            # empty lists here hides the fact that a usable candidate existed and
            # was rejected by a gate, which makes the gate impossible to debug.
            return QueryResponse(
                question=question,
                pln_query="",
                original_query=original_query,
                executed_query="",
                query_source="none",
                fallback_used=False,
                query_status="no_query",
                raw_proof="",
                sources=[],
                answer="I couldn't translate this question into a logical query.",
                qdrant_aligned_queries=qdrant_queries,
                execution_candidates=[],
                intent_mode=intent.mode.value,
                proof_status="unknown",
                proof_validated=True,
                rejection_reasons=self._rejection_summary(
                    candidate_rejections,
                    str(parse_result.metadata.get("provider_error", "")),
                ),
                context_retrieval_seconds=round(context_retrieval_seconds, 4),
                parse_query_seconds=round(parse_query_seconds, 4),
                reasoning_seconds=0.0,
                source_lookup_seconds=0.0,
                answer_generation_seconds=0.0,
            )

        # 3. Run reasoning read-only via PeTTaChainer against ordered candidates
        t2 = time.perf_counter()
        proof_traces: List[str] = []
        proof_outcome = ProofOutcome()
        first_outcome: ProofOutcome | None = None
        executed_query = ""
        executed_source = "none"
        candidates = self._proof_equivalent_candidates(
            candidate_entries,
            candidate_rejections,
        )
        if not self._query_fallback_enabled:
            candidates = candidates[:1]

        candidate_count_total = len(candidates)
        max_tries = int(getattr(cfg, "query_candidate_max_tries", 0) or 0)
        if self._query_fallback_enabled and max_tries > 0:
            candidates = candidates[:max_tries]
        candidate_count_tried = len(candidates)

        executed_candidate_index: int | None = None
        retry_used = False
        for idx, (candidate, source) in enumerate(candidates):
            executed_query = candidate
            executed_source = source
            executed_candidate_index = idx
            proof_outcome = self._prove_candidate(candidate, seed_terms)
            if first_outcome is None:
                first_outcome = proof_outcome
            proof_traces = proof_outcome.proof
            if proof_outcome.status != "unknown":
                break

        if not proof_traces and hasattr(self._parser, "retry_parse_query"):
            try:
                retry = getattr(self._parser, "retry_parse_query")
                retry_result = retry(question, context, executed_query)
                if retry_result and retry_result.queries:
                    retry_used = True
                    retry_entries, retry_rejections = self._query_candidates(
                        question,
                        [],
                        retry_result.queries,
                    )
                    candidate_rejections.extend(retry_rejections)
                    validated_retry = [candidate for candidate, _source in retry_entries]
                    more = (
                        validated_retry
                        if self._query_fallback_enabled
                        else validated_retry[:1]
                    )
                    candidate_count_total += len(more)
                    if self._query_fallback_enabled and max_tries > 0:
                        remaining = max_tries - candidate_count_tried
                        more = more[: max(remaining, 0)]
                    candidate_count_tried += len(more)
                    for idx, candidate in enumerate(
                        more, start=(executed_candidate_index or 0) + 1
                    ):
                        executed_query = candidate
                        executed_source = "parser"
                        executed_candidate_index = idx
                        proof_outcome = self._prove_candidate(candidate, seed_terms)
                        proof_traces = proof_outcome.proof
                        if proof_outcome.status != "unknown":
                            break
            except Exception as exc:
                print(f"[Service] retry_parse_query failed: {exc}")

        if not proof_traces and candidates:
            executed_query, executed_source = candidates[0]
            executed_candidate_index = 0
            proof_outcome = first_outcome or ProofOutcome(
                positive_query=executed_query,
                negative_query=self._reasoner.negated_query(executed_query),
            )

        reasoning_seconds = time.perf_counter() - t2

        raw_proof = str(
            {
                "positive": proof_outcome.positive_proof,
                "negative": proof_outcome.negative_proof,
            }
        )
        fallback_used = bool(
            executed_source == "parser"
            and executed_query
            and original_query
            and executed_query != original_query
        )
        query_status = self._classify_query_status(
            question,
            executed_query or original_query,
            fallback_used,
        )

        # 5. Reverse-lookup NL sources from proof atoms
        t3 = time.perf_counter()
        sources = self._extract_sources(
            proof_traces, max_atoms=cfg.source_lookup_max_atoms
        )
        source_lookup_seconds = time.perf_counter() - t3

        # 6. Generate natural language answer
        t4 = time.perf_counter()
        requirements: List[dict] = []
        if intent.mode == QuestionMode.EXPLANATION:
            requirements = self._reasoner.explain_requirements(
                set(intent.terms),
                entity=intent.entities[0] if intent.entities else "",
                direction=intent.direction,
            )
        if requirements:
            answer = self._answer_structured_intent(intent, requirements)
            answer_generation_seconds = time.perf_counter() - t4
        elif cfg.answer_generation_enabled:
            binding_rows = self._binding_rows()
            if binding_rows or self._query_has_goal_variables(executed_query):
                # Open question: report the proved values, not a yes/no verdict.
                answer = self._answer_gen.generate_open_answer(
                    question,
                    executed_query,
                    binding_rows,
                    truncated=bool(
                        getattr(getattr(self, "_last_bindings", None), "truncated", False)
                    ),
                )
            else:
                answer = self._answer_gen.generate_from_polarity(
                    question,
                    executed_query,
                    proof_outcome.status,
                    proof_outcome.positive_proof,
                    proof_outcome.negative_proof,
                    support_kind=proof_outcome.support_kind,
                )
            answer_generation_seconds = time.perf_counter() - t4
        else:
            answer = ""
            answer_generation_seconds = 0.0
        if not proof_traces and query_status == "weakly_aligned":
            answer = (
                "No proof was found. The generated query is only weakly aligned with the current "
                "knowledge base, so the failure may come from query shape mismatch or missing witness facts."
            )

        return QueryResponse(
            question=question,
            pln_query=executed_query,
            original_query=original_query,
            executed_query=executed_query,
            query_source=executed_source,
            fallback_used=fallback_used,
            query_status=query_status,
            raw_proof=raw_proof,
            sources=sources,
            answer=answer,
            qdrant_aligned_queries=qdrant_queries,
            execution_candidates=[candidate for candidate, _ in candidates],
            candidate_count=candidate_count_total,
            candidate_count_tried=candidate_count_tried,
            executed_candidate_index=executed_candidate_index,
            retry_used=retry_used,
            context_retrieval_seconds=round(context_retrieval_seconds, 4),
            parse_query_seconds=round(parse_query_seconds, 4),
            reasoning_seconds=round(reasoning_seconds, 4),
            source_lookup_seconds=round(source_lookup_seconds, 4),
            answer_generation_seconds=round(answer_generation_seconds, 4),
            intent_mode=intent.mode.value,
            proof_status=proof_outcome.status,
            negative_query=proof_outcome.negative_query,
            positive_proof=proof_outcome.positive_proof,
            negative_proof=proof_outcome.negative_proof,
            requirements=requirements,
            canonical_proposition=self._reasoner.grounded_query_atom(executed_query),
            target_alignment=("exact" if executed_source == "deterministic" else "compatible"),
            proof_validated=proof_outcome.proof_validated,
            support_kind=proof_outcome.support_kind,
            unresolved_mentions=self._ambiguous_mentions(parse_result.metadata),
            bindings=self._binding_rows(),
        )

    async def debug_query(self, question: str) -> DebugQueryResponse:
        cfg = get_settings()
        intent = parse_question_intent(question)
        if self._vector_store:
            context, qdrant_matches = self._qdrant_context_with_matches(
                question,
                top_k=max(
                    self._context_top_k,
                    int(getattr(cfg, "query_alignment_top_k", 0) or 0),
                ),
            )
        else:
            context, qdrant_matches = [], []
        alignment_matches = self._alignment_matches(qdrant_matches)
        qdrant_alignment = (
            build_aligned_queries(question, alignment_matches)
            if getattr(cfg, "query_alignment_enabled", True)
            else None
        )
        qdrant_queries = qdrant_alignment.queries if qdrant_alignment else []
        seed_terms = extract_forward_seed_terms(alignment_matches)
        context = self._enrich_context(context)

        debug_info = self._parser.debug_parse_query(question, context)
        langextract_info = debug_info.get("langextract_postprocessed", {})
        pln_candidates = debug_info.get("pln_canonicalized", [])
        supporting_statements = debug_info.get("supporting_statements", [])

        original_query = pln_candidates[0] if pln_candidates else (
            qdrant_queries[0] if qdrant_queries else ""
        )
        trusted_queries = self._deterministic_query_candidates(
            question,
            allow_open_targets=intent.mode == QuestionMode.OPEN,
        )
        candidate_entries, candidate_rejections = self._query_candidates(
            question,
            qdrant_queries,
            pln_candidates,
            trusted_queries,
        )
        if intent.mode in {QuestionMode.FACTORS, QuestionMode.SUFFICIENCY}:
            requirements = self._reasoner.explain_requirements(
                set(intent.terms),
                entity=intent.entities[0] if intent.entities else "",
                direction=intent.direction,
            )
            return DebugQueryResponse(
                question=question,
                context=context,
                qdrant_matches=qdrant_matches,
                qdrant_aligned_queries=qdrant_queries,
                execution_candidates=[],
                langextract_postprocessed=LangExtractQueryPostprocessed(
                    **langextract_info
                ),
                pln_canonicalized_queries=pln_candidates,
                supporting_statements=supporting_statements,
                executed_query="",
                query_source="none",
                fallback_used=False,
                query_status="well_aligned" if requirements else "weakly_aligned",
                proof="",
                sources=[],
                answer=self._answer_structured_intent(intent, requirements),
                intent_mode=intent.mode.value,
                proof_status="unanswered",
                proof_validated=True,
                requirements=requirements,
                candidate_rejections=candidate_rejections,
            )
        if not candidate_entries:
            # pln_canonicalized_queries must carry the real parser output. It used
            # to be hardcoded to [], which reported an intent-gate rejection as if
            # the postprocessor had produced nothing.
            return DebugQueryResponse(
                question=question,
                context=context,
                qdrant_matches=qdrant_matches,
                qdrant_aligned_queries=qdrant_queries,
                execution_candidates=[],
                langextract_postprocessed=LangExtractQueryPostprocessed(
                    **langextract_info
                ),
                pln_canonicalized_queries=pln_candidates,
                supporting_statements=supporting_statements,
                executed_query="",
                query_source="none",
                fallback_used=False,
                query_status="no_query",
                proof="",
                sources=[],
                answer="I couldn't translate this question into a logical query.",
                intent_mode=intent.mode.value,
                proof_status="unknown",
                proof_validated=True,
                rejection_reasons=self._rejection_summary(
                    candidate_rejections,
                    str(langextract_info.get("provider_error", "")),
                ),
                candidate_rejections=candidate_rejections,
            )

        candidates = self._proof_equivalent_candidates(
            candidate_entries,
            candidate_rejections,
        )
        if not self._query_fallback_enabled:
            candidates = candidates[:1]
        max_tries = int(getattr(cfg, "query_candidate_max_tries", 0) or 0)
        if self._query_fallback_enabled and max_tries > 0:
            candidates = candidates[:max_tries]

        executed_query = ""
        executed_source = "none"
        proof_traces: List[str] = []
        proof_outcome = ProofOutcome()
        first_outcome: ProofOutcome | None = None
        for candidate, source in candidates:
            executed_query = candidate
            executed_source = source
            proof_outcome = self._prove_candidate(candidate, seed_terms)
            if first_outcome is None:
                first_outcome = proof_outcome
            proof_traces = proof_outcome.proof
            if proof_outcome.status != "unknown":
                break

        if not proof_traces and candidates:
            executed_query, executed_source = candidates[0]
            proof_outcome = first_outcome or ProofOutcome(
                positive_query=executed_query,
                negative_query=self._reasoner.negated_query(executed_query),
            )

        fallback_used = bool(
            executed_source == "parser"
            and executed_query
            and original_query
            and executed_query != original_query
        )
        query_status = self._classify_query_status(
            question,
            executed_query or original_query,
            fallback_used,
        )
        sources = self._extract_sources(
            proof_traces, max_atoms=cfg.source_lookup_max_atoms
        )
        requirements: List[dict] = []
        if intent.mode == QuestionMode.EXPLANATION:
            requirements = self._reasoner.explain_requirements(
                set(intent.terms),
                entity=intent.entities[0] if intent.entities else "",
                direction=intent.direction,
            )
        if requirements:
            answer = self._answer_structured_intent(intent, requirements)
        elif cfg.answer_generation_enabled:
            binding_rows = self._binding_rows()
            if binding_rows or self._query_has_goal_variables(executed_query):
                # Open question: report the proved values, not a yes/no verdict.
                answer = self._answer_gen.generate_open_answer(
                    question,
                    executed_query,
                    binding_rows,
                    truncated=bool(
                        getattr(getattr(self, "_last_bindings", None), "truncated", False)
                    ),
                )
            else:
                answer = self._answer_gen.generate_from_polarity(
                    question,
                    executed_query,
                    proof_outcome.status,
                    proof_outcome.positive_proof,
                    proof_outcome.negative_proof,
                    support_kind=proof_outcome.support_kind,
                )
        else:
            answer = ""
        if not proof_traces and query_status == "weakly_aligned":
            answer = (
                "No proof was found. The generated query is only weakly aligned with the current "
                "knowledge base, so the failure may come from query shape mismatch or missing witness facts."
            )

        return DebugQueryResponse(
            question=question,
            context=context,
            qdrant_matches=qdrant_matches,
            qdrant_aligned_queries=qdrant_queries,
            execution_candidates=[candidate for candidate, _ in candidates],
            langextract_postprocessed=LangExtractQueryPostprocessed(
                **langextract_info
            ),
            pln_canonicalized_queries=pln_candidates,
            supporting_statements=supporting_statements,
            executed_query=executed_query,
            query_source=executed_source,
            fallback_used=fallback_used,
            query_status=query_status,
            proof=str(
                {
                    "positive": proof_outcome.positive_proof,
                    "negative": proof_outcome.negative_proof,
                }
            ),
            sources=sources,
            answer=answer,
            intent_mode=intent.mode.value,
            proof_status=proof_outcome.status,
            negative_query=proof_outcome.negative_query,
            positive_proof=proof_outcome.positive_proof,
            negative_proof=proof_outcome.negative_proof,
            requirements=requirements,
            canonical_proposition=self._reasoner.grounded_query_atom(executed_query),
            target_alignment=("exact" if executed_source == "deterministic" else "compatible"),
            proof_validated=proof_outcome.proof_validated,
            support_kind=proof_outcome.support_kind,
            unresolved_mentions=self._ambiguous_mentions(langextract_info),
            bindings=self._binding_rows(),
            candidate_rejections=candidate_rejections,
        )

    def _rejection_summary(
        self,
        rejections: List[dict],
        provider_error: str = "",
    ) -> List[str]:
        """Collapse per-candidate rejections into distinct gate names for /query."""
        stages: List[str] = []
        if provider_error:
            stages.extend(["query_generation_failed", provider_error])
        for rejection in rejections:
            stage = str(rejection.get("stage") or "unknown")
            if stage not in stages:
                stages.append(stage)
        if not stages:
            return ["no_query_candidates_produced"]
        return stages

    def _binding_rows(self) -> List[dict]:
        """Enumerated answers for an open question, each with its own proof."""
        outcome = getattr(self, "_last_bindings", None)
        if not outcome or not outcome.solutions:
            return []
        return [
            {
                "bindings": solution.bindings,
                "atom": solution.atom,
                "proof": solution.proof,
                "support_kind": solution.support_kind,
            }
            for solution in outcome.solutions
        ]

    def _ambiguous_mentions(self, metadata: dict) -> List[dict]:
        mention_data = metadata.get("mention_prepass", metadata)
        ambiguous = mention_data.get("ambiguous_pronouns", []) if isinstance(mention_data, dict) else []
        return [item for item in ambiguous if isinstance(item, dict)]

    def _prove_candidate(
        self,
        candidate: str,
        seed_terms: List[str] | None,
    ) -> ProofOutcome:
        """
        Prove a candidate and orient the result to the question that was asked.

        When the semantic gate judged a target to be the negation of the question
        — `(NotEligibleForRenewal dawit)` for "Is Dawit eligible for renewal?" —
        proving it establishes a "no", not a "yes". Without this the system reports
        a valid proof of the opposite of the question as a confident yes.
        """
        if self._query_has_goal_variables(candidate):
            # Open target: the question asks which values hold, so enumerate them.
            # A yes/no proof cannot carry that answer.
            binding_outcome = self._reasoner.query_bindings(candidate)
            self._last_bindings = binding_outcome
            return ProofOutcome(
                status=binding_outcome.status,
                positive_query=candidate,
                negative_query="",
                positive_proof=binding_outcome.proof,
                negative_proof=[],
                proof_validated=True,
                support_kind=binding_outcome.support_kind,
            )
        self._last_bindings = None
        outcome = self._reasoner.query_polarity(candidate, seed_terms=seed_terms)
        if candidate not in getattr(self, "_negation_targets", set()):
            return outcome
        if outcome.status == "positive":
            return replace(
                outcome,
                status="negative",
                support_kind="explicit_negative",
                positive_proof=[],
                negative_proof=outcome.positive_proof,
            )
        if outcome.status == "negative":
            return replace(
                outcome,
                status="positive",
                support_kind="entailed",
                positive_proof=outcome.negative_proof,
                negative_proof=[],
            )
        return outcome

    def _proof_equivalent_candidates(
        self,
        entries: List[tuple[str, str]],
        rejections: List[dict] | None = None,
    ) -> List[tuple[str, str]]:
        if not entries:
            return []
        if not getattr(self, "_proof_equivalent_only", False):
            # Off by default. Collapsing to the top candidate's exact head+args
            # discarded correct targets whenever a near-duplicate outranked them —
            # on P3 it dropped (EligibleForRenewal dawit) in favour of the
            # negation-lexicalized (NotEligibleForRenewal dawit), turning a right
            # answer into a wrong one. It also made query_candidate_max_tries inert.
            return entries
        first = parse_query_signature(entries[0][0])
        if not first:
            self._record_equivalence_drops(entries[1:], rejections, "top candidate is malformed")
            return entries[:1]
        equivalent: List[tuple[str, str]] = []
        dropped: List[tuple[str, str]] = []
        for entry in entries:
            signature = parse_query_signature(entry[0])
            if not signature:
                dropped.append(entry)
                continue
            if (
                signature["head"] == first["head"]
                and signature["args"] == first["args"]
            ):
                equivalent.append(entry)
            else:
                dropped.append(entry)
        if not equivalent:
            self._record_equivalence_drops(entries[1:], rejections, "no candidate matched the top signature")
            return entries[:1]
        self._record_equivalence_drops(
            dropped,
            rejections,
            f"differs from the top-ranked target ({first['head']} {' '.join(first['args'])})",
        )
        return equivalent

    def _record_equivalence_drops(
        self,
        dropped: List[tuple[str, str]],
        rejections: List[dict] | None,
        detail: str,
    ) -> None:
        if rejections is None:
            return
        rejections.extend(
            {
                "query": query,
                "source": source,
                "stage": "proof_equivalence",
                "detail": detail,
            }
            for query, source in dropped
        )

    def _answer_structured_intent(
        self,
        intent: QuestionIntent,
        explanations: List[dict],
    ) -> str:
        if intent.mode == QuestionMode.SUFFICIENCY:
            if not explanations:
                return (
                    "The knowledge base does not establish that the stated cause alone is "
                    "sufficient. A related fact is not a proof of causal sufficiency."
                )
            atoms = self._requirement_atoms(explanations)
            additional = [atom for atom, status in atoms if status != "positive"]
            if additional:
                return (
                    "No proof establishes sufficiency from that condition alone. "
                    "The relevant rules also require: " + ", ".join(additional[:8]) + "."
                )
            return (
                "The relevant rule requirements are proved, but this planner does not treat "
                "that as proof that the named condition alone is sufficient."
            )

        if not explanations:
            return "No proof-backed factors matching this question were found in the rules."
        atoms = self._requirement_atoms(explanations)
        if not atoms:
            return "Relevant rules were found, but they contain no inspectable factor premises."
        rendered = ", ".join(f"{atom} [{status}]" for atom, status in atoms[:12])
        if intent.mode == QuestionMode.EXPLANATION:
            return "The proof rules depend on these requirements: " + rendered + "."
        return "The relevant rules identify these requirements: " + rendered + "."

    def _requirement_atoms(self, explanations: List[dict]) -> List[tuple[str, str]]:
        atoms: List[tuple[str, str]] = []
        seen: set[str] = set()

        def visit(item: dict) -> None:
            atom = str(item.get("atom", ""))
            if atom and atom not in seen:
                seen.add(atom)
                atoms.append((atom, str(item.get("status", "unknown"))))
            for key in ("children", "options"):
                for child in item.get(key, []) or []:
                    if isinstance(child, dict):
                        visit(child)

        for explanation in explanations:
            for requirement in explanation.get("requirements", []) or []:
                if isinstance(requirement, dict):
                    visit(requirement)
        return atoms

    def _classify_query_status(
        self, question: str, original_query: str, fallback_used: bool
    ) -> str:
        if not original_query:
            return "no_query"
        if fallback_used:
            return "weakly_aligned"

        normalized = question.strip().lower()
        is_yes_no = normalized.startswith(
            (
                "is ",
                "are ",
                "was ",
                "were ",
                "does ",
                "do ",
                "did ",
                "can ",
                "could ",
                "may ",
                "might ",
                "must ",
                "shall ",
                "should ",
                "has ",
                "have ",
                "had ",
            )
        )
        has_variables = self._query_has_goal_variables(original_query)
        if is_yes_no and has_variables:
            return "weakly_aligned"
        return "well_aligned"

    def _query_has_goal_variables(self, query: str) -> bool:
        variables = set(re.findall(r"[$?][A-Za-z_][A-Za-z0-9_]*", query))
        return bool(variables - {"$prf", "$tv", "?prf", "?tv"})

    def _extract_sources(self, proof_traces: List[str], max_atoms: int = 30) -> List[str]:
        """
        Extract atom names from proof traces and reverse-lookup
        their NL source sentences from the vector store.
        """
        exact_sources = self._reasoner.sources_for_proof(proof_traces)
        if exact_sources:
            return exact_sources
        if max_atoms <= 0:
            return []

        atoms_to_search = set()
        if not self._vector_store:
            return []
        for trace in proof_traces:
            for match in re.findall(r"\([^()]+?\)", str(trace)):
                if "STV" not in match and len(match) >= 5:
                    atoms_to_search.add(match)

        sources = set()
        if len(atoms_to_search) > max_atoms:
            atoms_to_search = set(list(atoms_to_search)[:max_atoms])

        for atom_str in atoms_to_search:
            try:
                vector = self._vector_store.embed(atom_str)
                ctx, _ = self._vector_store.retrieve_context(atom_str, top_k=1)
                # retrieve_context returns atoms, not NL — direct search needed
                resp = self._vector_store._client.post(
                    f"{self._vector_store._qdrant}/collections"
                    f"/{self._vector_store._collection}/points/search",
                    json={"vector": vector, "limit": 1, "with_payload": True},
                )
                results = resp.json().get("result", [])
                if results and results[0].get("score", 0) > 0.6:
                    nl = results[0].get("payload", {}).get("nl")
                    if nl:
                        sources.add(nl)
            except Exception:
                pass

        return list(sources)

    #  Reset

    def reset(self, scope: str):
        if scope in ("all", "atomspace"):
            self._reasoner.reset()
            self._parser.reset(clear_registry=scope == "all")
        if scope in ("all", "vectordb") and self._vector_store:
            self._vector_store.reset()
        ledger = getattr(self, "_evidence_ledger", None)
        if scope == "all" and ledger:
            ledger.reset()

    def rebuild_indexes(self) -> dict:
        ledger = getattr(self, "_evidence_ledger", None)
        if not ledger:
            return {
                "atomspace_count": 0,
                "qdrant_indexed_count": 0,
                "pending_index_count": 0,
            }
        atomspace_count = self._rebuild_reasoner_from_ledger()
        indexed_count = 0
        if self._vector_store:
            self._vector_store.reset_evidence()
            ledger.requeue_indexes()
            while ledger.pending_count:
                indexed = self._flush_evidence_index()
                if indexed <= 0:
                    break
                indexed_count += indexed
        return {
            "atomspace_count": atomspace_count,
            "qdrant_indexed_count": indexed_count,
            "pending_index_count": ledger.pending_count,
        }

    #  Health

    def health(self) -> dict:
        ledger = getattr(self, "_evidence_ledger", None)
        return {
            "atomspace_size": self._reasoner.size,
            "vectordb_count": self._vector_store.count if self._vector_store else 0,
            "parser": self._parser.__class__.__name__,
            "evidence_document_count": ledger.document_count if ledger else 0,
            "evidence_claim_count": ledger.claim_count if ledger else 0,
            "pending_index_count": ledger.pending_count if ledger else 0,
        }

    def debug_qdrant(self, limit: int = 50) -> dict:
        ledger = getattr(self, "_evidence_ledger", None)
        if not self._vector_store:
            return {
                "enabled": False,
                "count": 0,
                "points": [],
                "predicate_count": 0,
                "predicate_points": [],
                "evidence_document_count": ledger.document_count if ledger else 0,
                "evidence_claim_count": ledger.claim_count if ledger else 0,
                "pending_index_count": ledger.pending_count if ledger else 0,
            }
        capped_limit = max(1, min(limit, 200))
        return {
            "enabled": True,
            "count": self._vector_store.count,
            "points": self._vector_store.list_points(limit=capped_limit),
            "predicate_count": self._vector_store.predicate_count,
            "predicate_points": self._vector_store.list_predicate_points(
                limit=capped_limit
            ),
            "evidence_document_count": ledger.document_count if ledger else 0,
            "evidence_claim_count": ledger.claim_count if ledger else 0,
            "pending_index_count": ledger.pending_count if ledger else 0,
        }
