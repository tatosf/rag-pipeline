# RAG Pipeline

FastAPI backend that ingests PDFs and answers questions over them with cited sources, using Mistral AI for embeddings and generation. All retrieval primitives (BM25, cosine, fusion, MMR) are hand-rolled in pure Python — no LangChain, no vector DB.

## Run it

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "MISTRAL_API_KEY=your_key" > .env
uvicorn app.main:app --reload
```

Open `http://localhost:8000`.

Endpoints: `GET /` (UI), `POST /ingest` (multipart PDF), `POST /query` (`{"query": "..."}`), `GET /docs` (OpenAPI).

## How it operates

```
                ingest                              query
 PDF ─► pypdf ─► chunk ─► embed ─► SQLite     user query
                                     │              │
                                     │              ▼
                                     │         analyze (LLM)
                                     │      intent / rewrite / refusal
                                     │              │
                                     ▼              ▼
                               ┌───────────────────────┐
                               │  BM25  +  cosine sim  │  top-20 each
                               └───────────┬───────────┘
                                           ▼
                                     RRF fusion
                                   + confidence boost
                                   + MMR (diversity)
                                           ▼
                                      top-5 chunks
                                           ▼
                                    generate (LLM)
                                           ▼
                                    verify  (LLM)
                                           ▼
                            answer + citations + unsupported spans
```

### Ingestion

1. Extract text per page with `pypdf`.
2. Chunk into ~800-char windows with 150-char overlap, split on sentence boundaries.
3. Embed each chunk in batches of 32 (`mistral-embed`, 1024-dim, L2-normalized).
4. **Confidence dedup**: if a new chunk has cosine > 0.95 against any existing chunk, increment that chunk's `confidence` counter instead of inserting. Inspired by the LLM Wiki v2 idea that repeated knowledge should strengthen, not duplicate.
5. Store in SQLite as `(id, doc_name, page, text, embedding BLOB, confidence, timestamps)`.

### Query

1. **Analyze** (1 LLM call, JSON mode): classifies intent (`factual` / `list` / `table` / `greeting`), detects refusal categories (PII / legal / medical), and rewrites the query for retrieval. A regex prefilter short-circuits obvious PII (SSN, credit card) with no LLM call.
2. **Refusal or greeting** → template response, return.
3. **Hybrid search**: BM25 top-20 + cosine top-20 → **Reciprocal Rank Fusion** (`k=60`) → confidence boost (`score × (1 + 0.1·log(1+conf))`) → **MMR** (`λ=0.7`) for diversity → top-5.
4. **Evidence gate**: if the top fused score is below `MIN_EVIDENCE_SCORE`, return "insufficient evidence" rather than hallucinate.
5. **Generate**: template chosen by intent, retrieved chunks injected as context tagged `[doc.pdf p.N]`. System rule: *"Only use facts from the provided context. Never invent."*
6. **Verify** (1 LLM call): checks each answer sentence against the context. Unsupported sentences are returned as a separate list so the UI can dim them — the answer stays visible, but the user sees what isn't grounded.

Normal path = 3 LLM calls (analyze + generate + verify). Greeting = 1. Refusal = 0.

## Why these choices

- **No vector DB**: at a few thousand chunks, a numpy matrix in memory plus `matrix @ query_vec` beats any network hop to a vector store. The whole search path is a single file.
- **Hybrid BM25 + embeddings**: BM25 nails exact terms (names, codes) that embeddings blur; embeddings catch paraphrases BM25 misses. RRF fuses them without tuning.
- **Confidence score**: a tiny nod to knowledge-base lifecycle — repeated passages surface slightly higher than one-off ones, without building the full wiki machinery.
- **Per-sentence verification**: answer-level "is this grounded?" is binary; sentence-level tells the user *which part* to trust.

## Layout

```
app/
├── main.py               FastAPI routes
├── config.py             Thresholds, models, paths
├── schemas.py            Pydantic models
├── llm.py                Mistral client wrapper
├── ingestion/            pdf.py, chunker.py, pipeline.py
├── retrieval/            bm25.py, cosine_similarity.py, fusion.py, search.py, store.py
├── query/analyze.py      Intent + rewrite + refusal (1 LLM call)
└── generation/           prompts.py, generate.py, verify.py
static/index.html         Vanilla-JS chat UI
data/rag.db               SQLite (gitignored)
```

## Libraries

`fastapi`, `uvicorn`, `python-multipart`, `mistralai`, `pypdf`, `numpy`, `python-dotenv`. Stdlib `sqlite3` for storage. Nothing else — no LangChain, LlamaIndex, FAISS, Chroma, or rank_bm25.