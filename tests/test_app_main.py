from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from nba_game_projections.app import db, main


def _use_isolated_db(monkeypatch, tmp_path):
    test_engine = create_engine(f"sqlite:///{tmp_path / 'app.db'}")
    monkeypatch.setattr(db, "engine", test_engine)
    monkeypatch.setattr(db, "SessionLocal", sessionmaker(bind=test_engine))
    monkeypatch.setattr(db.config, "MODEL_ARTIFACTS_DIR", tmp_path / "models")
    return test_engine


def test_health_check_returns_ok(monkeypatch, tmp_path):
    _use_isolated_db(monkeypatch, tmp_path)
    with TestClient(main.app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_startup_creates_model_artifacts_dir_and_db_file(monkeypatch, tmp_path):
    test_engine = _use_isolated_db(monkeypatch, tmp_path)
    with TestClient(main.app):
        assert (tmp_path / "app.db").exists()
        assert (tmp_path / "models").is_dir()
        # Table names come from whatever's registered against Base at import
        # time (T5.2's roster_overrides, T5.3's model_versions, ...) - just
        # confirm create_all() actually ran, not the exact table set.
        assert inspect(test_engine).get_table_names()


def test_root_serves_static_ui(monkeypatch, tmp_path):
    _use_isolated_db(monkeypatch, tmp_path)
    with TestClient(main.app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "NBA Game Projections" in response.text
