from typer.testing import CliRunner
from openworld_knowledgegraphs.cli import app

runner = CliRunner()

def test_cli_end_to_end(tmp_path):
    db = tmp_path / "owkg.db"
    d1 = tmp_path / "a.txt"; d1.write_text("RAG retrieves enterprise context.", encoding="utf-8")
    d2 = tmp_path / "b.txt"; d2.write_text("Knowledge Graphs support reasoning.", encoding="utf-8")
    # ingest
    res = runner.invoke(app, ["docs", "ingest", str(d1), str(d2), "--db", str(db)])
    assert res.exit_code == 0
    # build retriever
    art = tmp_path / "tfidf.pkl"
    res = runner.invoke(app, ["retriever", "build", str(art), "--db", str(db)])
    if res.exit_code != 0:
        print("STDOUT:", res.stdout)
        print("STDERR:", res.stderr)
        if res.exception:
            print("EXCEPTION:", res.exception)
    assert res.exit_code == 0
    # query
    res = runner.invoke(app, ["retriever", "query", "enterprise context", "--db", str(db), "--artifact", str(art), "--k", "2"])
    assert res.exit_code == 0
    # build KG
    res = runner.invoke(app, ["kg", "build-from-docs", "--db", str(db), "--min-count", "1"])
    assert res.exit_code == 0
    # neighbors (may be empty if no caps), still should succeed
    res = runner.invoke(app, ["kg", "neighbors", "RAG", "--db", str(db), "--k", "5"])
    assert res.exit_code == 0
    # agent dummy
    res = runner.invoke(app, ["agents", "ask", "Explain Graphs", "--db", str(db), "--artifact", str(art), "--backend", "dummy", "--k", "1"])
    assert res.exit_code == 0
