import re

def extract_all_terms(pln_string: str) -> list[str]:
    # Extract terms (arguments) that are not capitalized (predicates) and not STV/Implication, etc.
    # Simple heuristic: find everything in parentheses, take the atoms
    pass
