from app.generation.prompts import (
    factual_template,
    greeting_template,
    list_template,
    table_template,
)
from app.llm import chat
from app.retrieval.search import RetrievedChunk


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Render chunks with source tags the model can cite verbatim."""
    parts: list[str] = []
    for c in chunks:
        tag = f"[{c.doc_name} p.{c.page}]"
        parts.append(f"{tag}\n{c.text}")
    return "\n\n---\n\n".join(parts)


def generate_answer(query: str, chunks: list[RetrievedChunk], intent: str) -> str:
    if intent == "greeting":
        messages = greeting_template(query)
    else:
        context = _format_context(chunks)
        if intent == "list":
            messages = list_template(query, context)
        elif intent == "table":
            messages = table_template(query, context)
        else:
            messages = factual_template(query, context)
    return chat(messages, temperature=0.2)