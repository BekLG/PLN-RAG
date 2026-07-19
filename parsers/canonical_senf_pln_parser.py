from typing import List
from core.parser import ParseResult, SemanticParser
from parsers.canonical_pln_parser import CanonicalPLNParser
from core.senf import build_senf_from_atoms
from core.exemplar_registry import ExemplarScorer

class CanonicalSenfPlnParser(SemanticParser):
    """
    Phase 1 & 2 SENF Builder extension.
    Wraps CanonicalPLNParser to produce SENF objects and Exemplar scores,
    then serializes them back to MeTTa atoms.
    """
    def __init__(self):
        self.base_parser = CanonicalPLNParser()
        self.scorer = ExemplarScorer()

    def parse(self, text: str, context: List[str]) -> ParseResult:
        result = self.base_parser.parse(text, context)
        return self._enrich_result(result, text)

    def parse_batch(self, texts: List[str], context: List[str]) -> ParseResult:
        result = self.base_parser.parse_batch(texts, context)
        # Combine texts for context hints
        full_text = " ".join(texts)
        return self._enrich_result(result, full_text)

    def _enrich_result(self, result: ParseResult, context_text: str) -> ParseResult:
        # Enrich statements
        if result.statements:
            senf = build_senf_from_atoms(result.statements)
            self.scorer.score(senf, context_text)
            result.statements = senf.to_metta_strings()

        # Enrich queries
        if result.queries:
            query_senf = build_senf_from_atoms(result.queries)
            self.scorer.score(query_senf, context_text)
            result.queries = query_senf.to_metta_strings()

        return result
