from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.generation.generate import generate_answer
from app.generation.prompts import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    REFUSAL_MESSAGES,
    greeting_template,
)
from app.generation.verify import verify_answer
from app.ingestion.pipeline import IngestReport, ingest_pdf
from app.llm import chat
from app.query.analyze import analyze_query
from app.retrieval.search import hybrid_search
from app.schemas import (
    Citation,
    Health,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)

app = FastAPI(title="RAG Pipeline")

STATIC_DIR = Path("static")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/health", response_model=Health)
def health() -> Health:
    return Health(status="ok")


@app.get("/")
def root():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return {"message": "UI not present. See /docs for the API."}
    return FileResponse(str(index))


@app.post("/ingest", response_model=IngestResponse)
async def ingest(files: list[UploadFile] = File(...)) -> IngestResponse:
    max_bytes = settings.max_upload_mb * 1024 * 1024
    reports: list[IngestReport] = []
    for f in files:
        name = (f.filename or "unnamed.pdf").lower()
        ctype = (f.content_type or "").lower()
        if ctype != "application/pdf" and not name.endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Not a PDF: {f.filename}")
        data = await f.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"Empty file: {f.filename}")
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"'{f.filename}' exceeds {settings.max_upload_mb} MB limit",
            )
        reports.append(ingest_pdf(f.filename or "unnamed.pdf", data))
    return IngestResponse(results=reports)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    q = req.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query is empty")

    analysis = analyze_query(q)

    # 1. Refusal short-circuit
    if analysis.refusal:
        return QueryResponse(
            answer=REFUSAL_MESSAGES[analysis.refusal],
            intent=analysis.intent,
            refused=True,
            refusal_reason=analysis.refusal,
            insufficient_evidence=False,
            citations=[],
            unsupported_sentences=[],
            rewritten_query=None,
        )

    # 2. No-search short-circuit (greetings, meta)
    if not analysis.needs_search:
        answer = chat(greeting_template(q), temperature=0.4)
        return QueryResponse(
            answer=answer,
            intent=analysis.intent,
            refused=False,
            refusal_reason=None,
            insufficient_evidence=False,
            citations=[],
            unsupported_sentences=[],
            rewritten_query=None,
        )

    # 3. Hybrid search
    chunks, max_score = hybrid_search(analysis.rewritten_query)

    # 4. Evidence threshold gate
    if not chunks or max_score < settings.min_evidence_score:
        return QueryResponse(
            answer=INSUFFICIENT_EVIDENCE_MESSAGE,
            intent=analysis.intent,
            refused=False,
            refusal_reason=None,
            insufficient_evidence=True,
            citations=[],
            unsupported_sentences=[],
            rewritten_query=analysis.rewritten_query,
        )

    # 5. Generate, then verify
    answer = generate_answer(q, chunks, analysis.intent)
    unsupported = verify_answer(answer, chunks) if settings.enable_verify else []

    citations = [
        Citation(
            chunk_id=c.id,
            doc_name=c.doc_name,
            page=c.page,
            score=c.score,
            text=c.text,
        )
        for c in chunks
    ]

    return QueryResponse(
        answer=answer,
        intent=analysis.intent,
        refused=False,
        refusal_reason=None,
        insufficient_evidence=False,
        citations=citations,
        unsupported_sentences=unsupported,
        rewritten_query=analysis.rewritten_query,
    )