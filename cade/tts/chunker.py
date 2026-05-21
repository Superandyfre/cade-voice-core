"""Sentence chunking helpers for future streaming TTS."""

from __future__ import annotations


class SentenceChunker:
    """Incrementally split generated text into speakable chunks."""

    def __init__(self, min_chars: int = 20, max_chars: int = 160, comma_chars: int = 60):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.comma_chars = comma_chars
        self.buffer = ""

    def feed(self, delta: str) -> list[str]:
        self.buffer += str(delta or "")
        chunks: list[str] = []

        while True:
            cut = self._find_cut(self.buffer)
            if cut is None:
                break
            chunk = self.buffer[:cut].strip()
            self.buffer = self.buffer[cut:].lstrip()
            if chunk:
                chunks.append(chunk)

        if len(self.buffer) > self.max_chars:
            cut = self._last_space_before(self.buffer, self.max_chars)
            chunk = self.buffer[:cut].strip()
            self.buffer = self.buffer[cut:].lstrip()
            if chunk:
                chunks.append(chunk)

        return chunks

    def flush(self) -> list[str]:
        chunk = self.buffer.strip()
        self.buffer = ""
        return [chunk] if chunk else []

    def _find_cut(self, text: str) -> int | None:
        for i, char in enumerate(text):
            if char in ".?!":
                return i + 1
            if char == "," and i + 1 >= self.comma_chars:
                return i + 1
        return None

    @staticmethod
    def _last_space_before(text: str, limit: int) -> int:
        idx = text.rfind(" ", 0, limit)
        return idx if idx > 0 else limit
