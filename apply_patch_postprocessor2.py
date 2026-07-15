import re

def extract_terms(stmt):
    # remove parens
    s = stmt.replace("(", " ").replace(")", " ")
    tokens = s.split()
    terms = set()
    for t in tokens:
        # A valid term in atomspace is usually lowercase or CamelCase but we only want the actual concept objects, which are usually lowercase variables like `abebe`, `pasta`, `spaghetti`.
        if t.islower() and not re.match(r'^[0-9.]+$', t):
            if t not in {"and", "or", "not", "stv"}:
                terms.add(t)
    return terms

print(extract_terms("(IsA pasta carbohydrate) (STV 1.0 0.9)"))
