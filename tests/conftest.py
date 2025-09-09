import os
import pytest

@pytest.fixture(autouse=True)
def env_defaults(monkeypatch, tmp_path):
    # Use temp DB per test session to avoid interference
    dbp = tmp_path / "owkg.db"
    monkeypatch.setenv("OPENWORLDKG_DEFAULT_BACKEND", "dummy")
    monkeypatch.setenv("OPENWORLDKG_DB_PATH", str(dbp))
    os.makedirs(tmp_path, exist_ok=True)
