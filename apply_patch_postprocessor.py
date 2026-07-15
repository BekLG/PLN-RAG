import re

def apply_universal_identity(statements):
    """
    Extract all variable atoms (lowercase text) and create identity relationships.
    E.g., (IsA pasta carbohydrate) -> extract pasta, carbohydrate.
    Generate (IsA pasta pasta) and (IsA carbohydrate carbohydrate)
    """
    terms = set()
    for stmt in statements:
        # Very simple nested tokens extraction
        tokens = re.findall(r'\b[a-z0-9_]+\b', stmt)
        # remove stopwords or boolean operators
        for t in tokens:
            if t not in {"and", "or", "not", "isa", "stv", "implication"}:
                terms.add(t)

    new_stmts = []
    for t in terms:
        # Exclude things that look like float values for STV
        if not re.match(r'^\d', t):
            new_stmts.append(f"(IsA {t} {t})")
    
    return statements + new_stmts

# test
print(apply_universal_identity(["(Eats abebe pasta)"]))
