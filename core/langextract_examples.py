from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LangExtractPromptSpec:
    statement_prompt: str
    query_prompt: str
    statement_examples: list[Any]
    query_examples: list[Any]


def default_examples_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "langextract_examples.json"


@lru_cache(maxsize=8)
def load_langextract_prompt_spec(path: str | None = None) -> LangExtractPromptSpec:
    import langextract as lx

    if path:
        example_path = Path(path)
        if not example_path.is_absolute():
            example_path = Path(__file__).resolve().parents[1] / example_path
    else:
        example_path = default_examples_path()

    with example_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return LangExtractPromptSpec(
        statement_prompt=str(payload["statement_prompt"]),
        query_prompt=str(payload["query_prompt"]),
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
