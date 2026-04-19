import re
from dataclasses import dataclass

from app.llm import chat_json

_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CREDIT_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")


@dataclass
class QueryAnalysis:
    intent: str              # greeting | factual | list | table | other
    needs_search: bool
    refusal: str | None      # None | pii | legal | medical
    rewritten_query: str


_SYSTEM_PROMPT = """You are a query classifier for a RAG system over PDF documents.

Return a single JSON object with these fields:

- "intent": one of "greeting" | "factual" | "list" | "table" | "other"
    - "greeting": social pleasantries ("hello", "thanks")
    - "factual": asks for a single fact, definition, or explanation
    - "list": asks for a list of items ("what are the main ...", "list the ...")
    - "table": asks for a comparison or tabular data
    - "other": anything else
- "needs_search": true if answering requires the knowledge base; false if a direct reply is enough (greetings, meta questions about the assistant).
- "refusal": null, or one of:
    - "pii": query asks for sensitive personal info about identifiable people
    - "legal": query asks for legal advice
    - "medical": query asks for medical advice (dosages, diagnoses, treatments)
- "rewritten_query": a retrieval-optimized rewrite — expand acronyms, add clarifying context, drop filler. If needs_search is false, echo the original verbatim.

Return ONLY the JSON object, no prose."""


def _prefilter_pii(query: str) -> bool:
    return bool(_SSN.search(query) or _CREDIT_CARD.search(query))


def analyze_query(query: str) -> QueryAnalysis:
    if _prefilter_pii(query):
        return QueryAnalysis(
            intent="other",
            needs_search=False,
            refusal="pii",
            rewritten_query=query,
        )

    result = chat_json(messages=[
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ])

    intent = result.get("intent") or "other"
    if intent not in {"greeting", "factual", "list", "table", "other"}:
        intent = "other"

    refusal = result.get("refusal")
    if refusal not in {"pii", "legal", "medical", None}:
        refusal = None

    return QueryAnalysis(
        intent=intent,
        needs_search=bool(result.get("needs_search", True)),
        refusal=refusal,
        rewritten_query=result.get("rewritten_query") or query,
    )