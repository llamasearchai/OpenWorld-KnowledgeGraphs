from openworld_knowledgegraphs.agents.orchestrator import ask
from openworld_knowledgegraphs.data.docs import ingest_paths
from openworld_knowledgegraphs.rag.retriever import build_retriever
from openworld_knowledgegraphs.kg.store import KGStore, Triple

def test_dummy_agent_with_context(tmp_path):
    db = tmp_path / "owkg.db"
    p = tmp_path / "d.txt"
    p.write_text("Knowledge Graphs link Entities. Context helps answers.", encoding="utf-8")
    ingest_paths(str(db), [str(p)])
    art = tmp_path / "tfidf.pkl"
    build_retriever(str(db), str(art))
    # Seed KG
    kg = KGStore(str(db))
    kg.ensure_schema()
    kg.upsert_nodes([("Graphs", "Graphs"), ("Entities", "Entities")])
    kg.upsert_edges([Triple(s="Graphs", p="link", o="Entities")])
    out = ask("How do Graphs link Entities?", db_path=str(db), artifact_path=str(art), backend="dummy", k=1)
    assert out.startswith("DUMMY:")
