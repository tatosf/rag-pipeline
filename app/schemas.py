from pydantic import BaseModel, Field

from app.ingestion.pipeline import IngestReport


class Health(BaseModel):
    status: str


class Citation(BaseModel):
    chunk_id: str
    doc_name: str
    page: int
    score: float
    text: str


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)


class QueryResponse(BaseModel):
    answer: str
    intent: str
    refused: bool
    refusal_reason: str | None
    insufficient_evidence: bool
    citations: list[Citation]
    unsupported_sentences: list[str]
    rewritten_query: str | None


class IngestResponse(BaseModel):
    results: list[IngestReport]