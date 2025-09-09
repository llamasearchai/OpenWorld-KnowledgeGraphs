# OpenWorld-KnowledgeGraphs

OpenWorld-KnowledgeGraphs is a batteries-included Python project for building RAG + Knowledge Graph (KG) applications powered by LLM agents. It focuses on practical building blocks (ingestion, retrieval, KG storage, and agent orchestration) with clean interfaces and a strong developer experience.

Key capabilities:
- LLM backends: OpenAI SDK, Ollama, and the `llm` CLI (`llm-ollama`).
- RAG: Document ingestion and a simple TF‑IDF retriever (cosine similarity).
- KG: Lightweight construction with NetworkX and SQLite persistence.
- Agents: Orchestrator that blends top‑k retrieved docs with KG neighbors.
- Interfaces: FastAPI service, Typer CLI, sqlite-utils registry, and Datasette exploration.
- DX: `uv`, `hatch`, `tox`, `pytest`, `ruff`, `mypy`, `pre-commit`, GitHub Actions CI.

This repository is independent and illustrative, using synthetic/supplied text documents. It is not affiliated with any company.

Note on project status: The public API, CLI entry points, and module names described below are the intended interfaces. If something appears missing in your checkout, please open an issue or PR—contributions are welcome.

## Overview

OpenWorld-KnowledgeGraphs provides an end‑to‑end workflow:
- Ingest documents into SQLite.
- Build a TF‑IDF retriever artifact for simple, transparent RAG.
- Create a small KG from documents and query neighbors.
- Use an agent orchestrator to answer questions with both RAG context and KG neighbors.

## Architecture

- `data`: Document ingestion and export helpers.
- `rag`: TF‑IDF retriever build and query.
- `kg`: KG store (SQLite) + light extraction and neighbor queries.
- `agents`: Agent orchestrator with pluggable LLM backends (dummy/OpenAI/Ollama/llm).
- `api`: FastAPI app exposing health, RAG, and agent endpoints.
- `cli`: Typer CLI wrapping ingestion, retriever, KG, agent, and server commands.

## Repository Layout

- `src/openworld_knowledgegraphs/`: Python package (library + CLI + API).
- `tests/`: Pytest suite.
- `.github/workflows/ci.yml`: CI pipeline (lint, typecheck, tests with coverage).
- `pyproject.toml`: Project metadata, dependencies, scripts, and tool config.
- `.pre-commit-config.yaml`: Linting/type checks on commit.

## Installation

Prerequisites:
- Python 3.11+
- `uv` (recommended): `pip install uv`
- Optional: Ollama with a local model, e.g., `ollama pull llama3`
- Optional: OpenAI key in `OPENAI_API_KEY`
- Optional: `llm` CLI and `llm-ollama` plugin: `pip install llm llm-ollama`

Install locally (editable):
```bash
uv venv
uv pip install -e ".[dev]"
uv pip install -e ".[agents]"  # optional agents backends
pre-commit install
```

## Quickstart

Bootstrap a toy workspace:
```bash
# Create sample docs
mkdir -p data db artifacts
echo "Knowledge graphs link entities. RAG retrieves context." > data/doc1.txt
echo "Agents reason over graphs and text." > data/doc2.txt

# Ingest
owkg docs ingest --paths data/doc1.txt data/doc2.txt --db db/owkg.db

# Build retriever index
owkg retriever build --db db/owkg.db --artifact artifacts/tfidf.pkl

# Query RAG
owkg retriever query --db db/owkg.db --artifact artifacts/tfidf.pkl --question "What links entities?"

# Build KG from docs
owkg kg build-from-docs --db db/owkg.db --min-count 1

# Ask the agent
owkg agents ask --question "How do agents use graphs?" --backend dummy --db db/owkg.db --artifact artifacts/tfidf.pkl
```

Run the API locally:
```bash
owkg serve --db db/owkg.db --port 8000
# Examples:
# curl -X POST localhost:8000/rag/query -H "content-type: application/json" -d '{"question":"What is RAG?","k":2, "artifact":"artifacts/tfidf.pkl"}'
# curl -X POST localhost:8000/agents/ask -H "content-type: application/json" -d '{"question":"Explain graphs and RAG","backend":"dummy", "artifact":"artifacts/tfidf.pkl"}'
```

Explore the DB with Datasette:
```bash
owkg datasette serve --db db/owkg.db --port 8010
```

Notes:
- Tests use a deterministic Dummy LLM backend; network calls are not required.
- You can extend the KG extractor and/or integrate embeddings if desired.

## Configuration

Settings are provided via environment variables (alongside sensible defaults). A starter file is provided at `.env.example`. Common variables include:
- `OPENAI_API_KEY`: API key for OpenAI backend.
- `OPENKG_DB_PATH`: Default SQLite database path (e.g., `db/owkg.db`).
- `OPENKG_DEFAULT_BACKEND`: One of `dummy`, `openai`, `ollama`, or `llm`.
- `OPENKG_DEFAULT_MODEL`: Default model name (backend dependent).
- `OPENKG_API_HOST` / `OPENKG_API_PORT`: Bind address for the FastAPI server.

Load your own `.env` in development by copying and editing:
```bash
cp .env.example .env
```

## Development

- Format: `uv run ruff format .`
- Lint: `uv run ruff check .`
- Types: `uv run mypy src`
- Tests: `uv run pytest --cov=src --cov-report=term-missing`
- Serve API: `uv run uvicorn openworld_knowledgegraphs.api.app:app --reload --port 8000`

Pre-commit hooks are configured to keep style and types consistent:
```bash
pre-commit install
pre-commit run --all-files
```

GitHub Actions CI runs lint, typecheck, and tests (see `.github/workflows/ci.yml`).

## Versioning and Commits

We follow Semantic Versioning. For commit messages and PR titles, prefer Conventional Commits (e.g., `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`). This improves changelog generation and PR review.

Examples:
- `feat(rag): add n-gram tokenizer`
- `fix(kg): correct neighbor query for isolated nodes`
- `docs(readme): clarify quickstart steps`

## Contributing

Issues and PRs are welcome. See `CONTRIBUTING.md` for:
- Branching and PR process
- Commit conventions
- Development environment setup
- Testing and coverage

## License

MIT (see `LICENSE`).

## Acknowledgements

- Thanks to the Python, FastAPI, Typer, NetworkX, scikit-learn, Datasette, and sqlite-utils communities.
