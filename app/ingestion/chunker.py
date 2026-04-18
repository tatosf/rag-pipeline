import re

_SENTENCE_END = re.compile(r"[.!?]\s+|\n\n")


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks, preferring sentence boundaries.

    size/overlap are in characters (~200/~40 tokens at 4 chars/token).
    When possible, each chunk ends at a sentence boundary within the last
    25% of the window, so chunks don't cut mid-sentence.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            search_start = max(start + int(size * 0.75), start + 1)
            window = text[search_start:end]
            matches = list(_SENTENCE_END.finditer(window))
            if matches:
                end = search_start + matches[-1].end()
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks