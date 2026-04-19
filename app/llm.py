import json
from typing import Any

import numpy as np
from mistralai.client import Mistral
from app.config import settings

_client = Mistral(api_key=settings.mistral_api_key)

_EMBED_BATCH_SIZE = 32


def embed_batch(texts: list[str]) -> np.ndarray:
    """Return (N, D) float32 L2-normalized embedding matrix."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    vecs: list[list[float]] = []
    for i in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[i : i + _EMBED_BATCH_SIZE]
        resp = _client.embeddings.create(model=settings.embed_model, inputs=batch)
        vecs.extend([d.embedding for d in resp.data])
    mat = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def embed_one(text: str) -> np.ndarray:
    return embed_batch([text])[0]


def chat(
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> str:
    resp = _client.chat.complete(
        model=settings.chat_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def chat_json(
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 800,
) -> Any:
    """Chat completion forced to JSON output. Returns parsed dict (or {} on parse failure)."""
    resp = _client.chat.complete(
        model=settings.chat_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}