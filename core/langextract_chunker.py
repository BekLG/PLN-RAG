from __future__ import annotations

import re
from typing import List

from config import get_settings


_MERGE_CUES = {
    "it", "its", "they", "their", "them", "this", "these", "those", "he", "she", "his", "her",
    "therefore", "thus", "hence", "however", "so",
}
_MERGE_PREFIXES = (
    "this result", "this finding", "these findings", "this method", "this approach",
    "this model", "this suggests", "this indicates", "these results", "as a result", "for this reason",
)


class LangExtractChunker:
    """Paragraph-first chunker for the experimental LangExtract parser."""

    def __init__(self):
        cfg = get_settings()
        self.chunk_size = cfg.langextract_chunk_size or cfg.chunk_size

    def chunk(self, text: str) -> List[str]:
        return split_langextract_text(text, self.chunk_size)


def split_langextract_text(text: str, chunk_size: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            flush()
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip()]
            merged = _merge_sentence_groups(sentences)
            sub: list[str] = []
            sub_len = 0
            for sentence in merged:
                if sub and sub_len + len(sentence) > chunk_size:
                    chunks.append(" ".join(sub))
                    sub = [sentence]
                    sub_len = len(sentence)
                else:
                    sub.append(sentence)
                    sub_len += len(sentence) + 1
            if sub:
                chunks.append(" ".join(sub))
            continue

        if current_len + len(paragraph) > chunk_size and current:
            flush()
        current.append(paragraph)
        current_len += len(paragraph) + 2

    flush()
    return [chunk for chunk in chunks if chunk.strip()]


def _merge_sentence_groups(sentences: list[str]) -> list[str]:
    merged: list[str] = []
    for sentence in sentences:
        if merged and _should_merge_with_previous(sentence):
            merged[-1] = f"{merged[-1]} {sentence}".strip()
        else:
            merged.append(sentence)
    return merged


def _should_merge_with_previous(sentence: str) -> bool:
    normalized = sentence.strip().lower()
    if not normalized:
        return False
    tokens = normalized.split()
    if not tokens:
        return False
    if tokens[0] in _MERGE_CUES:
        return True
    prefix2 = " ".join(tokens[:2])
    if prefix2 in _MERGE_PREFIXES:
        return True
    prefix3 = " ".join(tokens[:3])
    return prefix3 in _MERGE_PREFIXES
