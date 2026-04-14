import asyncio
import re
import ast
from typing import List, Tuple

from config import get_settings
from core.chunker import Chunker
from core.parser import SemanticParser
from core.reasoner import Reasoner
from core.answer_generator import AnswerGenerator
from storage.vector_store import VectorStore
from api.models import IngestItemResult, QueryResponse


class PLNRAGService:
    """
    Orchestrates the full pipeline:
      Text → Chunker → Parser → Reasoner → AnswerGenerator

    This class is the only place that knows about all components.
    Each component only knows about its own interface.
    """

    def __init__(self, parser: SemanticParser):
        cfg = get_settings()
        self._parser = parser
        self._chunker = Chunker()
        self._reasoner = Reasoner()
        self._vector_store = VectorStore()
        self._answer_gen = AnswerGenerator()
        self._context_top_k = cfg.context_top_k

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

    def _ingest_single(self, text: str) -> IngestItemResult:
        try:
            # 1. Chunk large texts
            chunks = self._chunker.chunk(text)
            all_atoms: List[str] = []

            for chunk in chunks:
                # 2. Retrieve context from atomspace + vector store
                context, vector = self._vector_store.retrieve_context(
                    chunk, top_k=self._context_top_k
                )
                # Also supplement with recent atoms from disk
                context = self._enrich_context(context)

                # 3. Parse chunk → PLN atoms
                parse_result = self._parser.parse(chunk, context)

                if not parse_result.statements:
                    print(f"[Service] No statements for chunk: '{chunk[:60]}...'")
                    continue

                # 4. Add to atomspace via reasoner
                added = self._reasoner.add_statements(parse_result.statements)
                all_atoms.extend(added)

                # 5. Store in vector DB for future context retrieval
                if added:
                    self._vector_store.store(chunk, added, vector)

            return IngestItemResult(
                text=text,
                atoms=all_atoms,
                status="success"
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return IngestItemResult(
                text=text,
                status="failed",
                error=str(e)
            )

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

    async def query(self, question: str) -> QueryResponse:
        # 1. Retrieve context for translation
        context, _ = self._vector_store.retrieve_context(
            question, top_k=self._context_top_k
        )
        context = self._enrich_context(context)

        # 2. Parse question → PLN query
        parse_result = self._parser.parse(question, context)

        # Support Manhin parser's separate query mode
        if hasattr(self._parser, "parse_query"):
            parse_result = self._parser.parse_query(question, context)

        if not parse_result.queries:
            return QueryResponse(
                question=question,
                pln_query="",
                raw_proof="",
                sources=[],
                answer="I couldn't translate this question into a logical query."
            )

        target_query = parse_result.queries[0]

        # 3. Add any supporting statements the parser generated for the query
        if parse_result.statements:
            self._reasoner.add_statements(parse_result.statements)

        # 4. Run reasoning via PeTTaChainer
        proof_traces = self._reasoner.query(target_query)
        raw_proof = str(proof_traces)

        # 5. Reverse-lookup NL sources from proof atoms
        sources = self._extract_sources(proof_traces)

        # 6. Generate natural language answer
        answer = self._answer_gen.generate(question, proof_traces)

        return QueryResponse(
            question=question,
            pln_query=target_query,
            raw_proof=raw_proof,
            sources=sources,
            answer=answer
        )

    def _extract_sources(self, proof_traces: List[str]) -> List[str]:
        """
        Extract atom names from proof traces and reverse-lookup
        their NL source sentences from the vector store.
        """
        atoms_to_search = set()
        for trace in proof_traces:
            for match in re.findall(r'\([^()]+?\)', str(trace)):
                if "STV" not in match and len(match) >= 5:
                    atoms_to_search.add(match)

        sources = set()
        for atom_str in atoms_to_search:
            try:
                vector = self._vector_store.embed(atom_str)
                ctx, _ = self._vector_store.retrieve_context(atom_str, top_k=1)
                # retrieve_context returns atoms, not NL — direct search needed
                resp = self._vector_store._client.post(
                    f"{self._vector_store._qdrant}/collections"
                    f"/{self._vector_store._collection}/points/search",
                    json={"vector": vector, "limit": 1, "with_payload": True}
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
        if scope in ("all", "vectordb"):
            self._vector_store.reset()

    #  Health 

    def health(self) -> dict:
        return {
            "atomspace_size": self._reasoner.size,
            "vectordb_count": self._vector_store.count,
            "parser": self._parser.__class__.__name__,
        }
