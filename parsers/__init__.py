from core.parser import SemanticParser
from config import get_settings


def get_parser() -> SemanticParser:
    """
    Factory: returns the configured parser.
    Add new parsers here — the rest of the system never changes.
    """
    cfg = get_settings()
    name = cfg.parser.lower()

    if name == "nl2pln":
        from parsers.nl2pln_parser import NL2PLNParser

        return NL2PLNParser()

    if name == "canonical_pln":
        from parsers.canonical_pln_parser import CanonicalPLNParser

        return CanonicalPLNParser()

    if name == "manhin":
        from parsers.manhin_parser import ManhinParser

        return ManhinParser()

    if name == "langextract":
        from parsers.langextract_pln_parser import LangExtractPLNParser

        return LangExtractPLNParser()

    raise ValueError(
        f"Unknown parser '{name}'. Set PARSER to one of: nl2pln, canonical_pln, manhin, langextract"
    )
