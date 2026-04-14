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

    if name == "manhin":
        from parsers.manhin_parser import ManhinParser
        return ManhinParser()

    raise ValueError(
        f"Unknown parser '{name}'. "
        f"Set PARSER to one of: nl2pln, manhin"
    )
