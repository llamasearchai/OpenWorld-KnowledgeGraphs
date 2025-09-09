## Contributing to OpenWorld-KnowledgeGraphs

Thanks for your interest in contributing! This guide helps you set up a local environment, follow code style and testing practices, and open high‑quality PRs.

### Getting Started

- Use Python 3.11+.
- Install `uv` (recommended) with `pip install uv`.
- Create a virtual environment and install dependencies:
  - `uv venv`
  - `uv pip install -e ".[dev]"`
  - Optional backends: `uv pip install -e ".[agents]"`
- Install pre-commit hooks: `pre-commit install`.
- Install commit-msg hook: `pre-commit install --hook-type commit-msg`.
- Copy `.env.example` to `.env` and adjust local paths/keys as needed.

### Development Workflow

- Format: `uv run ruff format .`
- Lint: `uv run ruff check .`
- Types: `uv run mypy src`
- Tests: `uv run pytest --cov=src --cov-report=term-missing`
- Serve API for manual checks: `uv run uvicorn openworld_knowledgegraphs.api.app:app --reload --port 8000`

CI runs the same checks (see `.github/workflows/ci.yml`). Keep PRs green.

### Branching and PRs

- Create feature branches from `main` (e.g., `feat/rag-ngrams`, `fix/kg-neighbors`).
- Keep PRs focused and small; include tests and docs.
- Fill out the PR template and ensure the checklist passes.
- Link related issues (e.g., `Closes #123`).

### Conventional Commits

Use Conventional Commits in both commit messages and PR titles:

- Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`, `build`, `ci`.
- Scope is optional but encouraged: `feat(rag): add n-gram tokenizer`.
- Summary should be imperative and concise.

Benefits: easier PR review, automated changelogs, and readable history.

### Code Style and Quality

- Python: consistent formatting and imports via `ruff`.
- Type hints are required in new/changed code (mypy strict).
- Prefer small, pure functions; avoid unnecessary complexity.
- Add docstrings for public functions/classes and clarify assumptions.

### Testing

- Add targeted unit tests alongside new code.
- Keep tests fast and deterministic (use the dummy LLM backend unless explicitly testing integrations).
- Aim to preserve coverage thresholds configured in `pyproject.toml`.

### Documentation

- Update `README.md` when user-facing behavior changes.
- Add or update module/docstrings for new APIs.

### Release and Versioning

- We follow Semantic Versioning.
- Conventional Commits enable automated changelogs:
  - Generate changelog: `uv run hatch run changelog` or `uv run cz changelog`
  - Bump version + changelog: `uv run hatch run bump` or `uv run cz bump --changelog`

Thanks again for contributing and improving this project!
