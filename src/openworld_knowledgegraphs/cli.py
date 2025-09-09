from __future__ import annotations

import typer
from typing import List, Optional

from .config import settings
from .data.docs import ingest_paths
from .rag.retriever import build_retriever, query as retriever_query
from .rag.advanced_retriever import build_advanced_retriever, advanced_query
from .kg.store import KGStore, build_from_docs
from .kg.extractors import extract_knowledge
from .kg.queries import KGQuerier
from .agents.orchestrator import ask as agent_ask


app = typer.Typer(help="OpenWorld-KnowledgeGraphs CLI")


docs_app = typer.Typer(help="Document ingestion commands")
retriever_app = typer.Typer(help="Retriever commands")
kg_app = typer.Typer(help="Knowledge Graph commands")
agents_app = typer.Typer(help="Agent commands")
datasette_app = typer.Typer(help="Datasette helpers")

app.add_typer(docs_app, name="docs")
app.add_typer(retriever_app, name="retriever")
app.add_typer(kg_app, name="kg")
app.add_typer(agents_app, name="agents")
app.add_typer(datasette_app, name="datasette")


@docs_app.command("ingest")
def docs_ingest(
    paths: List[str] = typer.Argument(..., help="Paths to document files to ingest"),
    db: Optional[str] = typer.Option(None, help="Database path")
) -> None:
    dbp = db or settings.db_path
    n = ingest_paths(dbp, paths)
    typer.echo(f"ingested={n}")


@retriever_app.command("build")
def retriever_build(artifact: str, db: Optional[str] = typer.Option(None)) -> None:
    dbp = db or settings.db_path
    ap = build_retriever(dbp, artifact)
    typer.echo(ap)


@retriever_app.command("query")
def retriever_query_cmd(
    question: str = typer.Argument(..., help="Question to search for"),
    k: int = typer.Option(3, help="Number of results to return"),
    artifact: str = typer.Option(..., help="Path to retriever artifact"),
    db: Optional[str] = typer.Option(None, help="Database path"),
) -> None:
    dbp = db or settings.db_path
    res = retriever_query(dbp, artifact, question, k=k)
    typer.echo(str(res))


@retriever_app.command("build-advanced")
def retriever_build_advanced(
    artifact: str = typer.Argument(..., help="Path for advanced retriever artifact"),
    db: Optional[str] = typer.Option(None, help="Database path"),
    include_bm25: bool = typer.Option(True, help="Include BM25 scoring"),
) -> None:
    dbp = db or settings.db_path
    ap = build_advanced_retriever(dbp, artifact, include_bm25=include_bm25)
    typer.echo(f"Advanced retriever built: {ap}")


@retriever_app.command("query-advanced")
def retriever_query_advanced_cmd(
    question: str = typer.Argument(..., help="Question to search for"),
    k: int = typer.Option(3, help="Number of results to return"),
    artifact: str = typer.Option(..., help="Path to advanced retriever artifact"),
    method: str = typer.Option("hybrid", help="Retrieval method (tfidf, bm25, hybrid)"),
    db: Optional[str] = typer.Option(None, help="Database path"),
) -> None:
    dbp = db or settings.db_path
    res = advanced_query(dbp, artifact, question, k=k, method=method)
    for i, (doc_id, score, text) in enumerate(res, 1):
        typer.echo(f"{i}. [ID: {doc_id}, Score: {score:.4f}] {text[:100]}...")
        typer.echo("")


@kg_app.command("build-from-docs")
def kg_build_from_docs(min_count: int = 1, db: Optional[str] = typer.Option(None)) -> None:
    dbp = db or settings.db_path
    added = build_from_docs(dbp, min_count=min_count)
    typer.echo(f"triples_added={added}")


@kg_app.command("neighbors")
def kg_neighbors(
    node: str = typer.Argument(..., help="Node to find neighbors for"),
    k: int = typer.Option(5, help="Number of neighbors to return"),
    db: Optional[str] = typer.Option(None, help="Database path"),
) -> None:
    dbp = db or settings.db_path
    store = KGStore(dbp)
    store.ensure_schema()
    res = store.neighbors(node, k=k)
    typer.echo(str(res))


@kg_app.command("extract")
def kg_extract(
    text: str = typer.Argument(..., help="Text to extract knowledge from"),
    extractor: str = typer.Option("hybrid", help="Extractor type (regex, ner, hybrid)"),
) -> None:
    """Extract knowledge from text."""
    result = extract_knowledge(text, extractor_type=extractor)
    typer.echo(f"Extracted {len(result.triples)} triples:")
    for triple in result.triples:
        typer.echo(f"  {triple.s} --{triple.p}--> {triple.o}")
    typer.echo(f"\nEntities: {', '.join(result.entities)}")
    typer.echo(f"Relations: {', '.join(result.relations)}")


@kg_app.command("query")
def kg_query(
    query_type: str = typer.Argument(..., help="Query type (neighbors, paths, relation, similar)"),
    entity: str = typer.Argument(..., help="Primary entity for query"),
    target: Optional[str] = typer.Argument(None, help="Target entity (for paths query)"),
    relation: Optional[str] = typer.Option(None, help="Relation type to filter by"),
    depth: int = typer.Option(1, help="Maximum depth for neighbor queries"),
    k: int = typer.Option(10, help="Maximum number of results"),
    db: Optional[str] = typer.Option(None, help="Database path"),
) -> None:
    """Advanced knowledge graph querying."""
    dbp = db or settings.db_path
    store = KGStore(dbp)
    store.ensure_schema()
    querier = KGQuerier(store)
    
    if query_type == "neighbors":
        result = querier.find_neighbors(entity, max_depth=depth, k=k)
        typer.echo(f"Found {len(result.triples)} neighbor relationships:")
        for triple in result.triples:
            typer.echo(f"  {triple.s} --{triple.p}--> {triple.o}")
            
    elif query_type == "paths":
        if not target:
            typer.echo("Target entity required for paths query")
            return
        result = querier.find_paths(entity, target, max_path_length=depth, max_paths=k)
        typer.echo(f"Found {len(result.paths or [])} paths from {entity} to {target}:")
        for i, path in enumerate(result.paths or [], 1):
            path_str = " -> ".join([path[0].s] + [t.o for t in path])
            typer.echo(f"  Path {i}: {path_str}")
            
    elif query_type == "relation":
        if not relation:
            typer.echo("Relation type required for relation query")
            return
        result = querier.find_by_relation(relation, k=k)
        typer.echo(f"Found {len(result.triples)} triples with relation '{relation}':")
        for triple in result.triples:
            typer.echo(f"  {triple.s} --{triple.p}--> {triple.o}")
            
    elif query_type == "similar":
        result = querier.find_similar_entities(entity, k=k)
        typer.echo(f"Found {len(result.triples)} entities similar to '{entity}':")
        for triple in result.triples:
            score = result.scores.get(f"{triple.s}|{triple.p}|{triple.o}", 0.0)
            typer.echo(f"  {triple.o} (similarity: {score:.3f})")
    else:
        typer.echo(f"Unknown query type: {query_type}")


@kg_app.command("stats")
def kg_stats(
    entity: Optional[str] = typer.Argument(None, help="Specific entity to get stats for"),
    db: Optional[str] = typer.Option(None, help="Database path"),
) -> None:
    """Show knowledge graph statistics."""
    dbp = db or settings.db_path
    store = KGStore(dbp)
    store.ensure_schema()
    querier = KGQuerier(store)
    
    stats = querier.get_entity_stats(entity)
    
    if entity:
        typer.echo(f"Statistics for entity '{entity}':")
        typer.echo(f"  Total connections: {stats['total_connections']}")
        typer.echo(f"  Outgoing relations: {stats['outgoing_relations']}")
        typer.echo(f"  Incoming relations: {stats['incoming_relations']}")
    else:
        typer.echo("Knowledge Graph Statistics:")
        typer.echo(f"  Total triples: {stats['total_triples']}")
        typer.echo(f"  Total entities: {stats['total_entities']}")
        typer.echo(f"  Total relations: {stats['total_relations']}")
        typer.echo(f"  Top relations: {dict(list(stats['relation_counts'].items())[:5])}")


@agents_app.command("ask")
def agents_ask_cmd(
    question: str = typer.Argument(..., help="Question to ask the agent"),
    backend: str = typer.Option("dummy", help="LLM backend to use (dummy, openai, ollama, llm)"),
    k: int = typer.Option(1, help="Number of documents and neighbors to retrieve"),
    artifact: Optional[str] = typer.Option(None, help="Path to TF-IDF retriever artifact"),
    db: Optional[str] = typer.Option(None, help="Database path"),
    # Backend-specific options
    model: Optional[str] = typer.Option(None, help="Model name for the backend"),
    api_key: Optional[str] = typer.Option(None, help="API key for OpenAI backend"),
    host: Optional[str] = typer.Option(None, help="Host URL for Ollama backend"),
) -> None:
    dbp = db or settings.db_path
    if not artifact:
        raise typer.BadParameter("--artifact is required")
    
    # Build backend kwargs based on provided options
    backend_kwargs = {}
    if model:
        backend_kwargs["model"] = model
    elif backend == "openai":
        backend_kwargs["model"] = settings.openai_model
    elif backend == "ollama":
        backend_kwargs["model"] = settings.ollama_model
    elif backend == "llm":
        backend_kwargs["model"] = settings.llm_model
    
    if api_key:
        backend_kwargs["api_key"] = api_key
    elif backend == "openai":
        backend_kwargs["api_key"] = settings.openai_api_key
        
    if host:
        backend_kwargs["host"] = host
    elif backend == "ollama":
        backend_kwargs["host"] = settings.ollama_host
    
    out = agent_ask(
        question, 
        db_path=dbp, 
        artifact_path=artifact, 
        backend=backend, 
        k=k,
        **backend_kwargs
    )
    typer.echo(out)


@app.command("serve")
def serve(db: Optional[str] = typer.Option(None), port: int = 8000) -> None:
    # Import lazily to avoid heavy deps for basic commands
    import uvicorn  # type: ignore
    from .api.app import app as fastapi_app
    if db:
        # If user overrides db, update settings at runtime
        from .config import settings as st
        st.db_path = db  # type: ignore[attr-defined]
    uvicorn.run(fastapi_app, host="0.0.0.0", port=port)


@datasette_app.command("serve")
def datasette_serve(db: Optional[str] = typer.Option(None), port: int = 8010) -> None:
    import subprocess
    dbp = db or settings.db_path
    subprocess.run(["datasette", dbp, "-p", str(port)], check=True)


if __name__ == "__main__":
    app()
