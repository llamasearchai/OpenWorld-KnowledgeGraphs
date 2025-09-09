from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from ..config import settings
from ..rag import retriever as rag_retriever
from ..agents import orchestrator


app = FastAPI()


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


class RAGQuery(BaseModel):
    question: str
    k: int = 3
    artifact: str


@app.post("/rag/query")
def rag_query(q: RAGQuery):  # type: ignore[no-untyped-def]
    res = rag_retriever.query(settings.db_path, q.artifact, q.question, k=q.k)
    return {"results": [{"id": r[0], "score": r[1]} for r in res]}


class AgentAsk(BaseModel):
    question: str
    backend: str = "dummy"
    artifact: str
    k: int = 3


@app.post("/agents/ask")
def agents_ask(body: AgentAsk):  # type: ignore[no-untyped-def]
    text = orchestrator.ask(
        body.question,
        db_path=settings.db_path,
        artifact_path=body.artifact,
        backend=body.backend,
        k=body.k,
    )
    return {"text": text}

