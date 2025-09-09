from pathlib import Path
from openworld_knowledgegraphs.data.docs import ingest_paths, export_docs
from openworld_knowledgegraphs.rag.retriever import build_retriever, query as rquery

def test_ingest_and_retriever(tmp_path):
    db = tmp_path / "owkg.db"
    p1 = tmp_path / "d1.txt"
    p2 = tmp_path / "d2.txt"
    p1.write_text("Knowledge Graphs connect Entities. RAG retrieves context.", encoding="utf-8")
    p2.write_text("Agents reason over graphs and text.", encoding="utf-8")
    n = ingest_paths(str(db), [str(p1), str(p2)])
    assert n == 2
    df = export_docs(str(db))
    assert len(df) == 2
    art = tmp_path / "tfidf.pkl"
    build_retriever(str(db), str(art))
    res = rquery(str(db), str(art), "How do graphs connect entities?", k=2)
    assert len(res) <= 2
