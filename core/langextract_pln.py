from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)


_EXACT_ALIGNMENT_NAMES = {"MATCH_EXACT", "MATCH_GREATER", "MATCH_LESSER"}
_STATEMENT_CLASSES = {"fact", "type_decl", "property", "inheritance", "rule", "negation"}
_STRUCTURAL_HEADS = {"and": "And", ",": "And", "or": "Or", "not": "Not", "And": "And", "Or": "Or", "Not": "Not"}
_PLN_STRUCTURAL_HEADS = {"Implication", "Premises", "Conclusions", "STV", "And", "Or", "Not", "IsA"}
_SINGULAR_INVARIANT_SUFFIXES = ("ics", "ous", "ness", "ship", "ment")
_SINGULAR_INVARIANT_WORDS = {"series", "species", "rabies", "news", "physics", "mathematics", "economics", "electronics", "ethics", "politics"}


@dataclass
class TranslationReject:
    extraction_class: str
    extraction_text: str
    reason: str


@dataclass
class PLNExtractionTranslation:
    statements: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    rejected: list[TranslationReject] = field(default_factory=list)
    ctx: dict[str, Any] = field(default_factory=dict)


class PLNTranslationError(ValueError):
    pass


def build_canonicalization_context(source_text: str) -> dict[str, Any]:
    ctx: dict[str, Any] = {"proper": {}}
    if not source_text:
        return ctx
    sentence_starts: set[int] = set()
    for match in re.finditer(r"(?:^|[.!?]\s+)([A-Z][A-Za-z0-9_'-]*)", source_text):
        sentence_starts.add(match.start(1))
    occurrences: dict[str, list[tuple[str, bool]]] = {}
    for match in re.finditer(r"\b([A-Za-z][A-Za-z0-9_'-]*)\b", source_text):
        token = match.group(1)
        is_initial = match.start(1) in sentence_starts
        occurrences.setdefault(token.lower(), []).append((token, is_initial))
    proper: dict[str, str] = {}
    for lower, occs in occurrences.items():
        non_initial = [tok for tok, initial in occs if not initial]
        if non_initial and all(tok[0].isupper() for tok in non_initial):
            proper[lower] = non_initial[0]
    ctx["proper"] = proper
    return ctx


def translate_extractions_to_pln(extractions: Iterable[Any], *, source_text: str = "", skip_fuzzy: bool = True, truth_value: str = "(STV 1.0 1.0)") -> PLNExtractionTranslation:
    ctx = build_canonicalization_context(source_text)
    result = PLNExtractionTranslation(ctx=ctx)
    seen: set[str] = set()
    for index, ext in enumerate(extractions, start=1):
        cls = _ext_class(ext)
        text = _ext_text(ext)
        if skip_fuzzy and _has_fuzzy_alignment(ext):
            result.rejected.append(TranslationReject(cls, text, "fuzzy alignment"))
            continue
        ok, reason = is_safe_statement_extraction(ext)
        if not ok:
            result.rejected.append(TranslationReject(cls, text, reason))
            continue
        try:
            name, payload = _statement_payload_from_extraction(ext, index, ctx)
            statement = f"(: {name} {payload} {truth_value})"
        except Exception as exc:
            result.rejected.append(TranslationReject(cls, text, str(exc)))
            continue
        if statement in seen:
            continue
        seen.add(statement)
        result.statements.append(statement)
    return result


def translate_query_extractions_to_pln(extractions: Iterable[Any], *, source_text: str = "") -> PLNExtractionTranslation:
    ctx = build_canonicalization_context(source_text)
    result = PLNExtractionTranslation(ctx=ctx)
    seen_queries: set[str] = set()
    for ext in extractions:
        cls = _ext_class(ext)
        text = _ext_text(ext)
        try:
            if cls == "query":
                query = _query_from_extraction(ext, ctx)
                if query not in seen_queries:
                    seen_queries.add(query)
                    result.queries.append(query)
                continue
            if cls in _STATEMENT_CLASSES:
                _name, payload = _statement_payload_from_extraction(ext, 1, ctx)
                if payload.startswith("(Implication"):
                    result.rejected.append(TranslationReject(cls, text, "query cannot target a rule"))
                    continue
                query = f"(: $prf {payload} $tv)"
                if query not in seen_queries:
                    seen_queries.add(query)
                    result.queries.append(query)
                continue
            result.rejected.append(TranslationReject(cls, text, f"unsupported query class '{cls}'"))
        except Exception as exc:
            result.rejected.append(TranslationReject(cls, text, str(exc)))
    return result


def collect_predicate_heads(extractions: Iterable[Any]) -> list[str]:
    heads: list[str] = []
    seen: set[str] = set()
    for ext in extractions:
        attrs = _ext_attrs(ext)
        for key in ("predicate", "head_predicate"):
            value = attrs.get(key)
            if isinstance(value, str) and value.strip():
                head = value.strip().lower().replace(" ", "-")
                if head not in seen:
                    seen.add(head)
                    heads.append(head)
    return heads


def format_context_hint(context: list[str], predicate_heads: list[str]) -> str:
    heads = _dedupe_preserve_order(predicate_heads + _extract_context_predicates(context))
    if not heads:
        return ""
    pairs = ", ".join(f"{head}->{_kebab_symbol(head)}" for head in heads[:24])
    return (
        "\n\nExisting predicate vocabulary from previous chunks and the knowledge base:\n"
        f"{pairs}\n"
        "Reuse these predicate names when the sentence or question means the same relation."
    )


def log_rejections(prefix: str, rejected: list[TranslationReject]) -> None:
    for reject in rejected:
        logger.warning("%s rejected %s extraction %r: %s", prefix, reject.extraction_class, reject.extraction_text, reject.reason)


def is_safe_statement_extraction(ext: Any) -> tuple[bool, str]:
    cls = _ext_class(ext)
    attrs = _ext_attrs(ext)
    if cls not in _STATEMENT_CLASSES:
        return False, f"unsupported statement class '{cls}'"
    if cls in {"fact", "negation", "property", "type_decl", "inheritance"}:
        keys = ("subject", "object", "entity", "type", "value", "child", "parent")
        for key in keys:
            if key in attrs and _has_free_variable(attrs[key]):
                return False, f"free variable in '{key}' for {cls}"
    if cls == "rule":
        if not _has_nonempty(attrs, "head_predicate"):
            return False, "rule has empty head_predicate"
        if not _has_nonempty(attrs, "body"):
            return False, "rule has empty body"
        try:
            _parse_sexp(str(attrs["body"]))
        except Exception as exc:
            return False, f"rule body unparseable: {exc}"
    return True, ""


def _statement_payload_from_extraction(ext: Any, index: int, ctx: Optional[dict[str, Any]]) -> tuple[str, str]:
    cls = _ext_class(ext)
    attrs = _ext_attrs(ext)
    if cls == "fact":
        expr = _fact_expr(attrs)
        return _statement_name(expr, index, "fact", ctx), _translate_predicate_expr(expr, ctx)
    if cls == "type_decl":
        expr = ["isa", _required(attrs, "entity"), _required(attrs, "type")]
        return _statement_name(expr, index, "type", ctx), _translate_predicate_expr(expr, ctx)
    if cls == "property":
        expr = ["has", _required(attrs, "entity"), _required(attrs, "property"), _required(attrs, "value")]
        return _statement_name(expr, index, "property", ctx), _translate_predicate_expr(expr, ctx)
    if cls == "inheritance":
        expr = ["isa", _required(attrs, "child"), _required(attrs, "parent")]
        return _statement_name(expr, index, "inheritance", ctx), _translate_predicate_expr(expr, ctx)
    if cls == "negation":
        expr = _fact_expr(attrs)
        payload = f"(Not {_translate_predicate_expr(expr, ctx)})"
        return _statement_name(expr, index, "neg", ctx), payload
    if cls == "rule":
        head_predicate = _required(attrs, "head_predicate")
        head_args = _coerce_argument_list(attrs.get("head_args", "$x"), split_strings=True)
        head_expr = [head_predicate, *head_args]
        body_expr = _parse_sexp(_required(attrs, "body"))
        premises = _extract_rule_premises(body_expr, ctx)
        if not premises:
            raise PLNTranslationError("rule body produced no premises")
        premise_block = " ".join(premises)
        conclusion = _translate_predicate_expr(head_expr, ctx)
        payload = f"(Implication (Premises {premise_block}) (Conclusions {conclusion}))"
        return _statement_name(head_expr, index, "rule", ctx), payload
    raise PLNTranslationError(f"unsupported statement class '{cls}'")


def _query_from_extraction(ext: Any, ctx: Optional[dict[str, Any]]) -> str:
    attrs = _ext_attrs(ext)
    predicate = _required(attrs, "predicate")
    arguments = _coerce_argument_list(attrs.get("arguments"), split_strings=True)
    if not arguments:
        arguments = _fact_arguments(attrs)
    if not arguments:
        raise PLNTranslationError("query requires at least one argument")
    expr = [predicate, *arguments]
    return f"(: $prf {_translate_predicate_expr(expr, ctx)} $tv)"


def _fact_expr(attrs: dict[str, Any]) -> list[str]:
    predicate = _required(attrs, "predicate")
    arguments = _fact_arguments(attrs)
    if not arguments:
        raise PLNTranslationError("fact requires at least one argument")
    return [predicate, *arguments]


def _fact_arguments(attrs: dict[str, Any]) -> list[str]:
    if "arguments" in attrs:
        return _coerce_argument_list(attrs["arguments"])
    args: list[str] = []
    if _has_nonempty(attrs, "subject"):
        args.append(str(attrs["subject"]))
    if _has_nonempty(attrs, "object"):
        args.append(str(attrs["object"]))
    return args


def _extract_rule_premises(expr: Any, ctx: Optional[dict[str, Any]]) -> list[str]:
    if isinstance(expr, list) and len(expr) >= 4 and str(expr[0]) == "match" and str(expr[1]) == "&self":
        expr = expr[2]
    return _premise_items(expr, ctx)


def _premise_items(expr: Any, ctx: Optional[dict[str, Any]]) -> list[str]:
    if not isinstance(expr, list) or not expr:
        raise PLNTranslationError(f"rule premise must be an expression: {expr!r}")
    head = str(expr[0])
    if head in {",", "and", "And"}:
        premises: list[str] = []
        for child in expr[1:]:
            premises.extend(_premise_items(child, ctx))
        return premises
    if head in {"or", "Or", "not", "Not"}:
        return [_translate_structural_expr(expr, ctx)]
    return [_translate_predicate_expr(expr, ctx)]


def _translate_structural_expr(expr: Sequence[Any], ctx: Optional[dict[str, Any]]) -> str:
    if not expr:
        raise PLNTranslationError("structural expression cannot be empty")
    head = _STRUCTURAL_HEADS.get(str(expr[0]))
    if head is None:
        raise PLNTranslationError(f"unsupported structural head: {expr[0]}")
    children = " ".join(_translate_predicate_expr(child, ctx) for child in expr[1:])
    return f"({head} {children})"


def _translate_predicate_expr(expr: Any, ctx: Optional[dict[str, Any]]) -> str:
    if not isinstance(expr, list) or not expr:
        raise PLNTranslationError(f"predicate must be an S-expression: {expr!r}")
    raw_head = str(expr[0])
    if raw_head in {"not", "Not", "or", "Or", "and", "And", ","}:
        return _translate_structural_expr(expr, ctx)
    translated_head = _translate_head(raw_head, expr)
    args = " ".join(_translate_arg(arg, ctx) for arg in expr[1:])
    return f"({translated_head}{(' ' + args) if args else ''})"


def _translate_head(raw_head: str, expr: Sequence[Any]) -> str:
    if raw_head in {"isa", "Inheritance", "IsA", ":"} and len(expr) == 3:
        return "IsA"
    return _pascal_symbol(_normalize_predicate(raw_head))


def _translate_arg(arg: Any, ctx: Optional[dict[str, Any]]) -> str:
    if isinstance(arg, list):
        return _translate_predicate_expr(arg, ctx)
    token = str(arg).strip()
    if token.startswith("$") or token.startswith("?"):
        return "$" + _snake_symbol(token[1:], preserve_case=True)
    return _snake_symbol(_normalize_value(token, ctx=ctx))


def _statement_name(expr: Sequence[Any], index: int, suffix: str, ctx: Optional[dict[str, Any]]) -> str:
    if expr:
        head = _snake_symbol(_normalize_predicate(str(expr[0])))
        args = [_snake_symbol(_normalize_value(str(arg).lstrip("$?"), ctx=ctx)) for arg in expr[1:3] if not isinstance(arg, list)]
        parts = [part for part in args + [head, suffix] if part]
        return "_".join(parts)[:80] or f"{suffix}_{index}"
    return f"{suffix}_{index}"


def _normalize_predicate(value: str) -> str:
    text = value.strip()
    if not text:
        raise PLNTranslationError("predicate cannot be empty")
    return text.lower().replace(" ", "-")


def _normalize_value(value: str, *, ctx: Optional[dict[str, Any]]) -> str:
    text = value.strip()
    if not text:
        raise PLNTranslationError("symbol cannot be empty")
    parts = text.split()
    canonical = [_canonicalize_token(part, ctx) for part in parts]
    return "-".join(part for part in canonical if part)


def _canonicalize_token(token: str, ctx: Optional[dict[str, Any]]) -> str:
    if not token or token.startswith(("$", "?")):
        return token
    if "-" in token or any(ch.isdigit() for ch in token):
        return token
    lower = token.lower()
    if ctx and lower in ctx.get("proper", {}):
        return ctx["proper"][lower]
    return _singularize(lower)


def _singularize(word: str) -> str:
    if len(word) <= 3:
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


def _parse_sexp(text: str) -> Any:
    tokens = _tokenize(text)
    if not tokens:
        raise PLNTranslationError("empty expression")
    expr, pos = _parse_tokens(tokens, 0)
    if pos != len(tokens):
        raise PLNTranslationError("unexpected trailing tokens")
    return expr


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "()":
            tokens.append(ch)
            i += 1
            continue
        j = i
        while j < len(text) and not text[j].isspace() and text[j] not in "()":
            j += 1
        tokens.append(text[i:j])
        i = j
    return tokens


def _parse_tokens(tokens: list[str], pos: int) -> tuple[Any, int]:
    token = tokens[pos]
    if token != "(":
        return token, pos + 1
    pos += 1
    expr: list[Any] = []
    while pos < len(tokens) and tokens[pos] != ")":
        child, pos = _parse_tokens(tokens, pos)
        expr.append(child)
    if pos >= len(tokens):
        raise PLNTranslationError("missing closing parenthesis")
    return expr, pos + 1


def _has_free_variable(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().startswith(("$", "?"))
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return any(str(item).strip().startswith(("$", "?")) for item in value)
    return False


def _has_fuzzy_alignment(ext: Any) -> bool:
    status = getattr(ext, "alignment_status", None)
    if status is None:
        return False
    name = getattr(status, "name", None) or str(status).split(".")[-1]
    return name not in _EXACT_ALIGNMENT_NAMES


def _ext_class(ext: Any) -> str:
    return str(getattr(ext, "extraction_class", "")).strip()


def _ext_text(ext: Any) -> str:
    return str(getattr(ext, "extraction_text", "")).strip()


def _ext_attrs(ext: Any) -> dict[str, Any]:
    attrs = getattr(ext, "attributes", None) or {}
    return dict(attrs)


def _required(attrs: dict[str, Any], key: str) -> str:
    value = attrs.get(key)
    if value is None or not str(value).strip():
        raise PLNTranslationError(f"missing required attribute '{key}'")
    return str(value).strip()


def _has_nonempty(attrs: dict[str, Any], key: str) -> bool:
    return key in attrs and attrs[key] is not None and bool(str(attrs[key]).strip())


def _coerce_argument_list(value: Any, *, split_strings: bool = False) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if split_strings:
            if "," in text:
                return [part.strip() for part in text.split(",") if part.strip()]
            return [part for part in text.split() if part]
        return [text]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        result: list[str] = []
        for item in value:
            item_text = str(item).strip()
            if item_text:
                result.append(item_text)
        return result
    item_text = str(value).strip()
    return [item_text] if item_text else []


def _extract_context_predicates(context: list[str]) -> list[str]:
    seen: set[str] = set()
    heads: list[str] = []
    for atom in context:
        for match in re.findall(r"\(([A-Za-z][A-Za-z0-9_]*)", atom):
            if match in _PLN_STRUCTURAL_HEADS:
                continue
            if match not in seen:
                seen.add(match)
                heads.append(match)
    return heads


def _pascal_symbol(value: str) -> str:
    if value in {"IsA", "And", "Or", "Not"}:
        return value
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", value) if p]
    if not parts:
        return value
    return "".join(part[:1].upper() + part[1:].lower() for part in parts)


def _snake_symbol(value: str, *, preserve_case: bool = False) -> str:
    value = value.strip()
    if not value:
        return value
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    value = re.sub(r"[^A-Za-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value if preserve_case else value.lower()


def _kebab_symbol(value: str) -> str:
    return _snake_symbol(value).replace("_", "-")


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        clean = item.strip()
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result
