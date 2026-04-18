from pathlib import Path

import numpy as np
from pydantic import BaseModel

from app.config import UPLOAD_DIR, settings
from app.ingestion.chunker import chunk_text
from app.ingestion.pdf import extract_pages
from app.llm import embed_batch
from app.retrieval.store import (
    bump_confidence,
    insert_chunk,
    load_embedding_matrix,
)


class IngestReport(BaseModel):
    file: str
    pages: int
    chunks_new: int
    chunks_reinforced: int


def ingest_pdf(filename: str, pdf_bytes: bytes) -> IngestReport:
    safe_name = Path(filename).name
    (UPLOAD_DIR / safe_name).write_bytes(pdf_bytes)

    pages = extract_pages(pdf_bytes)
    records: list[tuple[int, str]] = []
    for page_num, text in pages:
        for piece in chunk_text(text, size=settings.chunk_size, overlap=settings.chunk_overlap):
            records.append((page_num, piece))

    if not records:
        return IngestReport(file=safe_name, pages=len(pages), chunks_new=0, chunks_reinforced=0)

    embeddings = embed_batch([r[1] for r in records])

    ids, matrix = load_embedding_matrix()
    new_count = 0
    reinforced = 0

    for (page_num, piece), vec in zip(records, embeddings):
        if matrix.shape[0] > 0:
            sims = matrix @ vec
            best_idx = int(np.argmax(sims))
            if sims[best_idx] >= settings.dedup_threshold:
                bump_confidence(ids[best_idx])
                reinforced += 1
                continue
        new_id = insert_chunk(safe_name, page_num, piece, vec)
        row = vec.reshape(1, -1)
        matrix = row if matrix.shape[0] == 0 else np.vstack([matrix, row])
        ids.append(new_id)
        new_count += 1

    return IngestReport(
        file=safe_name,
        pages=len(pages),
        chunks_new=new_count,
        chunks_reinforced=reinforced,
    )