from typing import List
from core.parser import ParseResult, SemanticParser
from parsers.canonical_pln_parser import CanonicalPLNParser
from core.senf import build_senf_from_atoms
from core.exemplar_registry import ExemplarScorer
from core.identity_graph import IdentityGraphBuilder
from core.transweave import TransWeaveAligner
from core.pln_bridge import PLNBridgeGenerator

class CanonicalSenfPlnParser(SemanticParser):
    def __init__(self):
        self.base_parser = CanonicalPLNParser()
        self.scorer = ExemplarScorer()
        self.identity_builder = IdentityGraphBuilder()
        self.aligner = TransWeaveAligner()
        self.bridge_generator = PLNBridgeGenerator()

    def parse(self, text: str, context: List[str]) -> ParseResult:
        result = self.base_parser.parse(text, context)
        return self._enrich_result(result, text, context)

    def parse_batch(self, texts: List[str], context: List[str]) -> ParseResult:
        result = self.base_parser.parse_batch(texts, context)
        full_text = " ".join(texts)
        return self._enrich_result(result, full_text, context)

    def _enrich_result(self, result: ParseResult, context_text: str, context_atoms: List[str]) -> ParseResult:
        # Build context SENF for TransWeave
        context_senf = None
        if context_atoms:
            context_senf = build_senf_from_atoms(context_atoms)
            self.scorer.score(context_senf, "")
            
        # Enrich statements
        if result.statements:
            senf = build_senf_from_atoms(result.statements)
            self.scorer.score(senf, context_text)
            self.identity_builder.build_graph(senf)
            
            # Phase 4: TransWeave against context if available
            if context_senf:
                weaves = self.aligner.build_weaves(context_senf, senf, weave_id="W_stmt", sa_id="Context", sb_id="Statement")
                for w in weaves:
                    senf.raw_atoms.extend(w.to_metta_strings())
                    # Phase 5: PLN Bridge Generation
                    bridge_atoms = self.bridge_generator.generate_bridges(w)
                    senf.raw_atoms.extend(bridge_atoms)
                    
            result.statements = senf.to_metta_strings()

        # Enrich queries
        if result.queries:
            query_senf = build_senf_from_atoms(result.queries)
            self.scorer.score(query_senf, context_text)
            self.identity_builder.build_graph(query_senf)
            
            # Phase 4: TransWeave against context if available
            if context_senf:
                weaves = self.aligner.build_weaves(context_senf, query_senf, weave_id="W_query", sa_id="Context", sb_id="Query")
                for w in weaves:
                    # Put the weaves and bridges into STATEMENTS so the reasoner loads them
                    if not result.statements:
                        result.statements = []
                    result.statements.extend(w.to_metta_strings())
                    # Phase 5: PLN Bridge Generation
                    result.statements.extend(self.bridge_generator.generate_bridges(w))

        return result

    def parse_query(self, text: str, context: List[str]) -> ParseResult:
        if hasattr(self.base_parser, "parse_query"):
            result = self.base_parser.parse_query(text, context)
        else:
            result = self.base_parser.parse(text, context)
        return self._enrich_result(result, text, context)
