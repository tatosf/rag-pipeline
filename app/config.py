import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "rag.db"
UPLOAD_DIR = DATA_DIR / "uploads"


@dataclass(frozen=True)
class Settings:
    mistral_api_key: str
    embed_model: str = "mistral-embed"
    chat_model: str = "mistral-small-latest"
    chunk_size: int = 800
    chunk_overlap: int = 150
    dedup_threshold: float = 0.95
    confidence_boost: float = 0.1
    top_k: int = 5
    candidate_k: int = 20
    min_evidence_score: float = 0.02
    enable_verify: bool = True
    max_upload_mb: int = 20


def load_settings() -> Settings:
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        raise RuntimeError(
            "MISTRAL_API_KEY not set. Copy .env.example to .env and fill in your key."
        )
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return Settings(mistral_api_key=key)


settings = load_settings()