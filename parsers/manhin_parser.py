from typing import List
from core.parser import SemanticParser, ParseResult


class ManhinParser(SemanticParser):
    """
    Manhin's LLM-based parser with format self-correction loop,
    FAISS predicate store, and equivalence generation.

    Expects the Manhin parser repo to be on PYTHONPATH or installed.
    Set PARSER=manhin in .env to activate.
    """

    def __init__(self):
        # Lazy import — only loaded if this parser is selected
        from pipelines import nl2pln as manhin_nl2pln
        from vector_index import faiss_store
        self._nl2pln = manhin_nl2pln
        self._faiss_store = faiss_store

    def parse(self, text: str, context: List[str]) -> ParseResult:
        try:
            # Build context dict in Manhin's expected format
            context_entries = [{"title": "Existing KB atoms", "content": "\n".join(context)}] if context else []

            result = self._nl2pln(text, context=context_entries, mode="parsing")
            if result is None:
                print(f"[ManhinParser] Failed for '{text}'")
                return ParseResult()

            _type_defs, stmts, _queries, extra_exprs, _sent_links = result

            # Merge stmts + extra_exprs as statements (type_defs excluded —
            # they are ontological and managed separately if needed)
            all_stmts = list(dict.fromkeys(stmts + extra_exprs))

            return ParseResult(statements=all_stmts, queries=[])

        except Exception as e:
            print(f"[ManhinParser] Exception for '{text}': {e}")
            return ParseResult()

    def parse_query(self, text: str, context: List[str]) -> ParseResult:
        """For question parsing — uses Manhin's querying mode."""
        try:
            context_entries = [{"title": "Existing KB atoms", "content": "\n".join(context)}] if context else []
            result = self._nl2pln(text, context=context_entries, mode="querying")
            if result is None:
                return ParseResult()
            _type_defs, stmts, queries, extra_exprs, _sent_links = result
            return ParseResult(
                statements=list(dict.fromkeys(stmts + extra_exprs)),
                queries=queries
            )
        except Exception as e:
            print(f"[ManhinParser] Query exception for '{text}': {e}")
            return ParseResult()
