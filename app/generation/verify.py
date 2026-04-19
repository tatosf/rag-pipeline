import re

from app.llm import chat_json
from app.retrieval.search import RetrievedChunk

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")

_SYSTEM = (
    "You check whether each sentence of an answer is supported by the provided context. "
    "Return a single JSON object with a 'sentences' array. Each element is "
    '{"sentence": str, "supported": bool}. '
    "A sentence is 'supported' only if the context clearly contains the claim it makes. "
    "Social phrases like 'Certainly.' or 'Here are the main points:' are always supported. "
    "Return ONLY the JSON, no prose."
)


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def verify_answer(answer: str, chunks: list[RetrievedChunk]) -> list[str]:
    """Return the sentences flagged as unsupported by the retrieved context."""
    sentences = _split_sentences(answer)
    if not sentences or not chunks:
        return []

    context = "\n\n---\n\n".join(f"[{c.doc_name} p.{c.page}] {c.text}" for c in chunks)

    result = chat_json(messages=[
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Answer:\n{answer}\n\n"
                f"Sentences (pre-split): {sentences}"
            ),
        },
    ])

    unsupported: list[str] = []
    for item in result.get("sentences", []):
        if isinstance(item, dict) and item.get("supported") is False:
            unsupported.append(str(item.get("sentence", "")))
    return unsupported