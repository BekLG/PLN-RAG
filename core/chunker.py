import re
from typing import List
from config import get_settings


class Chunker:
    """
    Processing layer: splits large text into parser-sized chunks.

    Runs BEFORE the semantic parser. The parser only ever sees
    clean, bounded input — never a raw document dump.
    """

    def __init__(self):
        cfg = get_settings()
        self.chunk_size = cfg.chunk_size
        self.overlap = cfg.chunk_overlap
        self._merge_cues = {
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
            "therefore",
            "thus",
            "hence",
            "however",
            "so",
        }
        self._merge_prefixes = (
            "this result",
            "this finding",
            "these findings",
            "this method",
            "this approach",
            "this model",
            "this suggests",
            "this indicates",
            "these results",
        )

    def chunk(self, text: str) -> List[str]:
        """
        Split text into sentence-first chunks.
        Merge adjacent sentences only when the next sentence appears to
        depend on the previous one. Oversized chunks still fall back to
        size-based splitting.
        """
        text = text.strip()
        if not text:
            return []

        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return self._split_oversized_chunk(text)

        merged: List[str] = []
        for sentence in sentences:
            if merged and self._should_merge_with_previous(sentence):
                merged[-1] = f"{merged[-1]} {sentence}".strip()
            else:
                merged.append(sentence)

        chunks: List[str] = []
        for chunk in merged:
            chunks.extend(self._split_oversized_chunk(chunk))
        return [chunk for chunk in chunks if chunk]

    def batch_chunks(self, text: str, max_sentences: int, max_chars: int) -> List[List[str]]:
        """
        Group sentence-first chunks into bounded parser batches.
        Each batch preserves chunk order while reducing LLM round-trips.
        """
        units = self.chunk(text)
        if not units:
            return []

        max_sentences = max(1, max_sentences)
        max_chars = max(1, max_chars)
        batches: List[List[str]] = []
        current: List[str] = []
        current_chars = 0

        for unit in units:
            unit_len = len(unit)
            exceeds_sentences = len(current) >= max_sentences
            exceeds_chars = current and (current_chars + 1 + unit_len) > max_chars
            if current and (exceeds_sentences or exceeds_chars):
                batches.append(current)
                current = []
                current_chars = 0

            current.append(unit)
            current_chars = current_chars + unit_len if current_chars == 0 else current_chars + 1 + unit_len

        if current:
            batches.append(current)
        return batches

    def _split_sentences(self, text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [part.strip() for part in parts if part.strip()]

    def _should_merge_with_previous(self, sentence: str) -> bool:
        normalized = sentence.strip().lower()
        if not normalized:
            return False
        tokens = normalized.split()
        if not tokens:
            return False
        if tokens[0] in self._merge_cues:
            return True
        prefix = " ".join(tokens[:2])
        if prefix in self._merge_prefixes:
            return True
        prefix = " ".join(tokens[:3])
        return prefix in self._merge_prefixes

    def _split_oversized_chunk(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size

            if end >= len(text):
                chunks.append(text[start:].strip())
                break

            # Try to break at a sentence boundary within the last 20% of chunk
            boundary = self._find_sentence_boundary(text, end)
            chunks.append(text[start:boundary].strip())
            start = boundary - self.overlap

        return [c for c in chunks if c]

    def _find_sentence_boundary(self, text: str, near: int) -> int:
        """Find the nearest sentence-ending punctuation before `near`."""
        search_from = max(0, near - int(self.chunk_size * 0.2))
        for i in range(near, search_from, -1):
            if text[i] in ".!?\n":
                return i + 1
        return near  # hard cut if no boundary found
