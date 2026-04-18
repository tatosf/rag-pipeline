import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    doc_name TEXT NOT NULL,
    page INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    confidence INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_name);
"""


@dataclass
class Chunk:
    id: str
    doc_name: str
    page: int
    text: str
    confidence: int


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def _from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


@contextmanager
def _connect(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(_SCHEMA)


def insert_chunk(doc_name: str, page: int, text: str, embedding: np.ndarray) -> str:
    chunk_id = str(uuid.uuid4())
    now = _now()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO chunks "
            "(id, doc_name, page, text, embedding, confidence, created_at, last_seen_at) "
            "VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
            (chunk_id, doc_name, page, text, _to_blob(embedding), now, now),
        )
    return chunk_id


def bump_confidence(chunk_id: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE chunks SET confidence = confidence + 1, last_seen_at = ? WHERE id = ?",
            (_now(), chunk_id),
        )


def all_chunks() -> list[Chunk]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, doc_name, page, text, confidence FROM chunks"
        ).fetchall()
    return [Chunk(*r) for r in rows]


def load_embedding_matrix() -> tuple[list[str], np.ndarray]:
    """Load all embeddings into a single (N, D) matrix. Returns (ids, matrix)."""
    with _connect() as conn:
        rows = conn.execute("SELECT id, embedding FROM chunks").fetchall()
    if not rows:
        return [], np.zeros((0, 0), dtype=np.float32)
    ids = [r[0] for r in rows]
    matrix = np.stack([_from_blob(r[1]) for r in rows], axis=0)
    return ids, matrix


def get_chunks_by_ids(ids: list[str]) -> dict[str, Chunk]:
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT id, doc_name, page, text, confidence FROM chunks WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
    return {r[0]: Chunk(*r) for r in rows}


init_db()