from openworld_knowledgegraphs.config import settings

def test_settings_defaults():
    assert settings.default_backend == "dummy"
    assert settings.db_path.endswith(".db")
