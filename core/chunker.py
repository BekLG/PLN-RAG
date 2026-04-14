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

    def chunk(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks at sentence boundaries.
        Falls back to hard character split if no sentence boundary found.
        """
        text = text.strip()
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
