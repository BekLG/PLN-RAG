from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


PROOF_SAFETY_PROMPT = """

Proof-safety rules:
- Preserve epistemic status: not known, not reported, not diagnosed, and not
  classified are not direct negations of the underlying property. Negate a
  status predicate such as clinically-classified-as-obese instead.
- Preserve hedging. For rules containing tend to, likely, or probably, include
  numeric strength and confidence attributes below 1.0.
- Never invent a premise, conclusion, entity, or relation that is not supported
  by an exact source span.
- Treat predicates used by rules in the same input as a document-local schema.
  When a later case sentence directly satisfies a rule premise, reuse that
  premise predicate and argument order instead of emitting a generic Has fact.
- Keep comparisons explicit in rule predicates: at-least, above, below, and
  duration/count requirements must not be reduced to equality.
- Keep compound entities and identifiers together, such as Lake Aster, Field
  Delta, Machine M7, Batch Q4, and Vendor BrightWare.
- Resolve a pronoun or possessive to an entity only when the antecedent is
  unambiguous. Attach status, event, and negation facts to that entity.
- A negative case fact may satisfy only a negative rule premise. Never turn
  "can", "may", "evidence against", or "alone is insufficient" into a
  deterministic positive or negative conclusion.
"""

QUERY_SAFETY_PROMPT = """

Query-safety rules:
- Reuse the exact document-local predicate whose proposition the question asks.
- Preserve compound entities and identifiers as one argument.
- Do not answer a positive capability/state question with a denial, prevention,
  failure, missing, invalid, or expired predicate.
- If no predicate expresses the requested proposition, return no extraction
  instead of selecting a merely related predicate.
"""


@dataclass(frozen=True)
class LangExtractPromptSpec:
    statement_prompt: str
    query_prompt: str
    statement_examples: list[Any]
    query_examples: list[Any]


def default_examples_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "langextract_examples.json"


@lru_cache(maxsize=8)
def load_langextract_prompt_spec(path: str | None = None) -> LangExtractPromptSpec:
    """Load LangExtract prompts/examples from PLN-RAG data files."""
    import langextract as lx

    if path:
        example_path = Path(path)
        if not example_path.is_absolute():
            example_path = Path(__file__).resolve().parents[2] / example_path
    else:
        example_path = default_examples_path()
    with example_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return LangExtractPromptSpec(
        statement_prompt=str(payload["statement_prompt"]) + PROOF_SAFETY_PROMPT,
        query_prompt=str(payload["query_prompt"]) + QUERY_SAFETY_PROMPT,
        statement_examples=_build_examples(lx, payload.get("statement_examples", [])),
        query_examples=_build_examples(lx, payload.get("query_examples", [])),
    )


def _build_examples(lx: Any, examples: list[dict[str, Any]]) -> list[Any]:
    built: list[Any] = []
    for example in examples:
        built.append(
            lx.data.ExampleData(
                text=str(example["text"]),
                extractions=[
                    lx.data.Extraction(
                        str(extraction["class"]),
                        str(extraction["text"]),
                        attributes=dict(extraction.get("attributes", {})),
                    )
                    for extraction in example.get("extractions", [])
                ],
            )
        )
    return built
