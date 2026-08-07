"""
Single source of truth for symbol canonicalization.

Ingestion and query planning must canonicalize the same word the same way, or an
atom stored through one path becomes unreachable from the other. This module used
to have a twin in `core/extraction/langextract_pln.py` that disagreed with it
(`series` -> `sery` here, `series` there), so both now delegate here.

Scientific and product identifiers are deliberately exempt from camel splitting
and lemmatization. The old camelCase rule turned `HbA1c` into `hb_a1c`, `IoT`
into `io_t`, and `mRNA` into `m_rna`, which then failed every downstream
comparison. Detection is structural — digits, uppercase runs, very short tokens —
rather than a word list, because maintaining a list of every identifier in every
domain is not a strategy.
"""

import re


# Bumped when canonicalization changes atom identity. v1 mangled identifiers and
# had two divergent singularizers.
NORMALIZATION_VERSION = 2

# Kept from the previous `langextract_pln` implementation, which was the more
# complete of the two. Do not grow these: reach for the semantic query gate
# instead of encoding more English here.
_SINGULAR_INVARIANT_SUFFIXES = ("ics", "ous", "ness", "ship", "ment")
_SINGULAR_INVARIANT_WORDS = frozenset({
    "series",
    "species",
    "rabies",
    "news",
    "physics",
    "mathematics",
    "economics",
    "electronics",
    "ethics",
    "politics",
})

_UPPERCASE_RUN = re.compile(r"[A-Z]{2,}")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SEPARATORS = re.compile(r"[^A-Za-z0-9]+")


def is_identifier_like(token: str) -> bool:
    """
    True when a token should keep its shape instead of being split and lemmatized.

    Covers `HbA1c` and `Cas9` (digits), `IoT` and `mRNA` (uppercase runs, or short
    enough that splitting is more likely to destroy than clarify). Ordinary
    multiword camel case such as `shutdownController` is not identifier-like and
    still splits.
    """
    token = token.strip()
    if not token:
        return False
    if any(char.isdigit() for char in token):
        return True
    if _UPPERCASE_RUN.search(token):
        return True
    return len(token) <= 3


def singularize(word: str) -> str:
    """Conservative English singularization. Identifier-like tokens pass through."""
    if len(word) <= 3 or is_identifier_like(word):
        return word
    if word in _SINGULAR_INVARIANT_WORDS:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ses") and len(word) > 4:
        return word[:-2]
    if word.endswith(_SINGULAR_INVARIANT_SUFFIXES):
        return word
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def pluralize(word: str) -> str:
    if word.endswith("y") and len(word) > 2:
        return word[:-1] + "ies"
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    return word + "s"


def split_symbol_parts(token: str) -> list[str]:
    """
    Split a raw token into canonical parts.

    Separators are applied first so that each part is classified on its own:
    `IoT-based` becomes `IoT` + `based`, letting the identifier stay intact while
    the ordinary word is still handled normally.
    """
    parts: list[str] = []
    for chunk in _SEPARATORS.split(token):
        if not chunk:
            continue
        if is_identifier_like(chunk):
            parts.append(chunk)
            continue
        parts.extend(piece for piece in _CAMEL_BOUNDARY.split(chunk) if piece)
    return parts


def canonical_symbol(token: str, lemmatize: bool = True, protect: bool = False) -> str:
    token = token.strip()
    if not token:
        return token
    parts = split_symbol_parts(token)
    if not parts:
        return ""
    if lemmatize and not protect:
        parts = [singularize(part.lower()) for part in parts]
    else:
        parts = [part.lower() for part in parts]
    return "_".join(part for part in parts if part)


def normalize_text(text: str) -> str:
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())
