from fastapi import FastAPI

from app.schemas import Health

app = FastAPI(title="RAG Pipeline")


@app.get("/health", response_model=Health)
def health() -> Health:
    return Health(status="ok")