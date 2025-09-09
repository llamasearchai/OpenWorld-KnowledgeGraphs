from openworld_knowledgegraphs.kg.store import KGStore, Triple, build_from_docs

def test_kg_store_edges_nodes(tmp_path):
    db = tmp_path / "owkg.db"
    kg = KGStore(str(db))
    kg.ensure_schema()
    n_nodes = kg.upsert_nodes([("SAP", "SAP"), ("AI", "AI")])
    assert n_nodes == 2
    n_edges = kg.upsert_edges([Triple(s="SAP", p="builds", o="AI")])
    assert n_edges == 1
    neigh = kg.neighbors("SAP", k=5)
    assert neigh and neigh[0].o == "AI"

def test_build_from_docs(tmp_path):
    db = tmp_path / "owkg.db"
    # Add a tiny doc
    from openworld_knowledgegraphs.data.docs import ingest_paths
    p = tmp_path / "d.txt"
    p.write_text("SAP builds AI. Agents improve Products.", encoding="utf-8")
    ingest_paths(str(db), [str(p)])
    added = build_from_docs(str(db), min_count=1)
    assert added >= 1
