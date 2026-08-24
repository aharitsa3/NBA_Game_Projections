from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from nba_game_projections.app import db, main
from nba_game_projections.app.routers import predictions


def _use_isolated_db(monkeypatch, tmp_path):
    test_engine = create_engine(f"sqlite:///{tmp_path / 'app.db'}")
    monkeypatch.setattr(db, "engine", test_engine)
    monkeypatch.setattr(db, "SessionLocal", sessionmaker(bind=test_engine))
    monkeypatch.setattr(db.config, "MODEL_ARTIFACTS_DIR", tmp_path / "models")


def test_get_slate_returns_predict_slate_result(monkeypatch, tmp_path):
    _use_isolated_db(monkeypatch, tmp_path)
    monkeypatch.setattr(
        predictions,
        "predict_slate",
        lambda session, date=None: [{"game_id": 1, "date": date, "predicted_margin": 3.5}],
    )

    with TestClient(main.app) as client:
        response = client.get("/predictions/slate", params={"date": "2027-01-15"})

    assert response.status_code == 200
    assert response.json() == [{"game_id": 1, "date": "2027-01-15", "predicted_margin": 3.5}]


def test_get_slate_returns_409_when_no_active_model(monkeypatch, tmp_path):
    _use_isolated_db(monkeypatch, tmp_path)

    def raise_no_active_model(session, date=None):
        raise predictions.NoActiveModelError("no active model")

    monkeypatch.setattr(predictions, "predict_slate", raise_no_active_model)

    with TestClient(main.app) as client:
        response = client.get("/predictions/slate")

    assert response.status_code == 409
