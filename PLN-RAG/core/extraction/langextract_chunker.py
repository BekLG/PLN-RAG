from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from config import get_settings


_MERGE_CUES = {
    "it",
    "its",
    "they",
    "their",
    "them",
    "this",
    "these",
    "those",
    "he",
    "she",
    "his",
    "her",
    "therefore",
    "thus",
    "hence",
    "however",
    "so",
}
_MERGE_PREFIXES = (
    "this result",
    "this finding",
    "these findings",
    "this method",
    "this approach",
    "this model",
    "this suggests",
    "this indicates",
    "these results",
    "as a result",
    "for this reason",
)


@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str
    start: int
    end: int


class LangExtractChunker:
    """
    LangExtract-style chunker ported from the standalone lang-extract project.

    It is paragraph-first, then sentence/coreference aware. PLN-RAG keeps its
    existing Chunker for other parsers.
    """

    def __init__(self):
        cfg = get_settings()
        self.chunk_size = cfg.langextract_chunk_size or cfg.chunk_size

    def chunk(self, text: str) -> List[str]:
        return split_langextract_text(text, self.chunk_size)

    def chunk_with_spans(self, text: str) -> list[TextChunk]:
        return split_langextract_text_with_spans(text, self.chunk_size)


def split_langextract_text(text: str, chunk_size: int) -> list[str]:
    """Split text near chunk_size respecting paragraph and sentence boundaries."""
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
            sentences = [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+", paragraph)
                if s.strip()
            ]
            merged = _merge_sentence_groups(sentences)
            sub: list[str] = []
            sub_len = 0
            for sentence in merged:
                if sub_len + len(sentence) > chunk_size and sub:
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


def split_langextract_text_with_spans(text: str, chunk_size: int) -> list[TextChunk]:
    """Split text while preserving offsets into the original document."""
    if not text or not text.strip():
        return []
    doc_start, doc_end = _trim_span(text, 0, len(text))
    if doc_end <= doc_start:
        return []
    if doc_end - doc_start <= chunk_size:
        return [TextChunk(0, text[doc_start:doc_end], doc_start, doc_end)]

    chunks: list[tuple[int, int]] = []
    current_start: int | None = None
    current_end: int | None = None

    def flush() -> None:
        nonlocal current_start, current_end
        if current_start is not None and current_end is not None:
            start, end = _trim_span(text, current_start, current_end)
            if end > start:
                chunks.append((start, end))
        current_start = None
        current_end = None

    for paragraph_start, paragraph_end in _paragraph_spans(text, doc_start, doc_end):
        paragraph_len = paragraph_end - paragraph_start
        if paragraph_len > chunk_size:
            flush()
            sentence_groups = _merge_sentence_span_groups(
                _sentence_spans(text, paragraph_start, paragraph_end),
                text,
            )
            sub_start: int | None = None
            sub_end: int | None = None
            for sentence_start, sentence_end in sentence_groups:
                if (
                    sub_start is not None
                    and sub_end is not None
                    and sentence_end - sub_start > chunk_size
                ):
                    chunks.append((sub_start, sub_end))
                    sub_start = sentence_start
                    sub_end = sentence_end
                else:
                    sub_start = sentence_start if sub_start is None else sub_start
                    sub_end = sentence_end
            if sub_start is not None and sub_end is not None:
                chunks.append((sub_start, sub_end))
            continue

        if (
            current_start is not None
            and current_end is not None
            and paragraph_end - current_start > chunk_size
        ):
            flush()
        current_start = paragraph_start if current_start is None else current_start
        current_end = paragraph_end

    flush()
    return [
        TextChunk(index=index, text=text[start:end], start=start, end=end)
        for index, (start, end) in enumerate(chunks)
        if text[start:end].strip()
    ]


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


def _paragraph_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = start
    for match in re.finditer(r"\n\s*\n", text[start:end]):
        raw_start = cursor
        raw_end = start + match.start()
        trimmed = _trim_span(text, raw_start, raw_end)
        if trimmed[1] > trimmed[0]:
            spans.append(trimmed)
        cursor = start + match.end()
    trimmed = _trim_span(text, cursor, end)
    if trimmed[1] > trimmed[0]:
        spans.append(trimmed)
    return spans


def _sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = start
    for match in re.finditer(r"[.!?](?:\s+|$)", text[start:end]):
        raw_end = start + match.end()
        trimmed = _trim_span(text, cursor, raw_end)
        if trimmed[1] > trimmed[0]:
            spans.append(trimmed)
        cursor = raw_end
    trimmed = _trim_span(text, cursor, end)
    if trimmed[1] > trimmed[0]:
        spans.append(trimmed)
    return spans


def _merge_sentence_span_groups(
    spans: list[tuple[int, int]],
    text: str,
) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        sentence = text[start:end]
        if merged and _should_merge_with_previous(sentence):
            previous_start, _previous_end = merged[-1]
            merged[-1] = (previous_start, end)
        else:
            merged.append((start, end))
    return merged


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end
