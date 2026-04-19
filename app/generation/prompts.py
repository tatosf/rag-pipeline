"""Prompt templates keyed by intent, plus canned messages for refusals and gaps."""

SYSTEM_BASE = (
    "You are a careful assistant that answers questions using ONLY the provided context. "
    "If the context does not contain the answer, say so explicitly. Never invent facts. "
    "Cite each claim inline using the format [doc.pdf p.X] immediately after the claim."
)


def factual_template(query: str, context: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_BASE + " Reply with a concise prose answer."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]


def list_template(query: str, context: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": SYSTEM_BASE
            + " Reply with a markdown bullet list. Each bullet carries its own citation.",
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]


def table_template(query: str, context: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": SYSTEM_BASE
            + " Reply with a markdown table. Put citations in the last column.",
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]


def greeting_template(query: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a friendly assistant for a PDF knowledge base. "
                "Reply briefly to social messages. Do not invent document content."
            ),
        },
        {"role": "user", "content": query},
    ]


INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I don't have enough supporting evidence in the provided documents to answer that. "
    "Try rephrasing your question, or check that the relevant PDFs have been ingested."
)


REFUSAL_MESSAGES: dict[str, str] = {
    "pii": (
        "I can't process queries that involve sharing personal identifying information "
        "(SSNs, credit card numbers, etc.)."
    ),
    "legal": (
        "This looks like a request for legal advice, which I'm not able to give. "
        "Please consult a qualified attorney. I can still share factual content from "
        "the ingested documents if you rephrase your question."
    ),
    "medical": (
        "This looks like a request for medical advice, which I'm not able to give. "
        "Please consult a qualified healthcare professional. I can still share factual "
        "content from the ingested documents if you rephrase your question."
    ),
}