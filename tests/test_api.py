from fastapi.testclient import TestClient
from openworld_knowledgegraphs.api.app import app
from openworld_knowledgegraphs.data.docs import ingest_paths
from openworld_knowledgegraphs.rag.retriever import build_retriever

def test_api_endpoints(tmp_path, monkeypatch):
    # Override settings.db_path via env var used by settings during import time is tricky,
    # but endpoints read settings once; instead, ingest/build into default path.
    # We simulate default path by using actual default file.
    import os
    db = tmp_path / "owkg.db"
    monkeypatch.setenv("OPENWORLDKG_DB_PATH", str(db))
    # Re-import settings to refresh
    from importlib import reload
    import openworld_knowledgegraphs.config as cfg
    reload(cfg)  # type: ignore
    from openworld_knowledgegraphs.config import settings as st
    # Seed docs and retriever
    p = tmp_path / "d.txt"; p.write_text("Agents use Graphs. RAG provides context.", encoding="utf-8")
    ingest_paths(st.db_path, [str(p)])
    art = tmp_path / "tfidf.pkl"
    build_retriever(st.db_path, str(art))

    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    r2 = client.post("/rag/query", json={"question": "context", "k": 1, "artifact": str(art)})
    assert r2.status_code == 200
    r3 = client.post("/agents/ask", json={"question": "Explain Graphs", "backend": "dummy", "artifact": str(art)})
    assert r3.status_code == 200
    assert r3.json()["text"].startswith("DUMMY:")
