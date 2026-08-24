from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from nba_game_projections.app import db, main
from nba_game_projections.app.repositories.predictions import build_prediction_repository
from nba_game_projections.app.routers import admin
from nba_game_projections.app.services.prediction import NoActiveModelError


def _use_isolated_db(monkeypatch, tmp_path):
    test_engine = create_engine(f"sqlite:///{tmp_path / 'app.db'}")
    test_sessionmaker = sessionmaker(bind=test_engine)
    monkeypatch.setattr(db, "engine", test_engine)
    monkeypatch.setattr(db, "SessionLocal", test_sessionmaker)
    monkeypatch.setattr(db.config, "MODEL_ARTIFACTS_DIR", tmp_path / "models")
    return test_sessionmaker


def _prediction_kwargs(**overrides) -> dict:
    fields = {
        "game_id": 1,
        "date": "2027-01-15",
        "home_team_id": 1,
        "home_team_name": "Team One",
        "away_team_id": 2,
        "away_team_name": "Team Two",
        "predicted_margin": 3.5,
        "predicted_winner": "home",
        "home_win_probability": 0.6,
        "model_version_id": 10,
    }
    fields.update(overrides)
    return fields


def test_capture_predictions_returns_serialized_records(monkeypatch, tmp_path):
    test_sessionmaker = _use_isolated_db(monkeypatch, tmp_path)

    with TestClient(main.app) as client:
        with test_sessionmaker() as session:
            record = build_prediction_repository(session).upsert_prediction(**_prediction_kwargs())
        monkeypatch.setattr(
            admin, "capture_pending_predictions", lambda session, horizon_days=14: [record]
        )

        response = client.post("/admin/predictions/capture")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["game_id"] == 1
    assert body[0]["reconciled_at"] is None


def test_capture_predictions_returns_409_when_no_active_model(monkeypatch, tmp_path):
    _use_isolated_db(monkeypatch, tmp_path)

    def raise_no_active_model(session, horizon_days=14):
        raise NoActiveModelError("no active model")

    monkeypatch.setattr(admin, "capture_pending_predictions", raise_no_active_model)

    with TestClient(main.app) as client:
        response = client.post("/admin/predictions/capture")

    assert response.status_code == 409


def test_capture_predictions_passes_horizon_days_query_param_through(monkeypatch, tmp_path):
    _use_isolated_db(monkeypatch, tmp_path)
    received = {}

    def fake_capture(session, horizon_days=14):
        received["horizon_days"] = horizon_days
        return []

    monkeypatch.setattr(admin, "capture_pending_predictions", fake_capture)

    with TestClient(main.app) as client:
        response = client.post("/admin/predictions/capture", params={"horizon_days": 30})

    assert response.status_code == 200
    assert received["horizon_days"] == 30


def test_capture_predictions_rejects_horizon_days_out_of_bounds(monkeypatch, tmp_path):
    _use_isolated_db(monkeypatch, tmp_path)

    with TestClient(main.app) as client:
        response = client.post("/admin/predictions/capture", params={"horizon_days": 90})

    assert response.status_code == 422


def test_reconcile_predictions_returns_serialized_records(monkeypatch, tmp_path):
    _use_isolated_db(monkeypatch, tmp_path)
    monkeypatch.setattr(admin, "reconcile_predictions", lambda session: [])

    with TestClient(main.app) as client:
        response = client.post("/admin/predictions/reconcile")

    assert response.status_code == 200
    assert response.json() == []


def test_get_prediction_accuracy_returns_service_result(monkeypatch, tmp_path):
    _use_isolated_db(monkeypatch, tmp_path)
    fake_report = {"pooled": {"n_games": 1}, "by_model_version": []}
    monkeypatch.setattr(admin, "compute_accuracy", lambda session: fake_report)

    with TestClient(main.app) as client:
        response = client.get("/admin/predictions/accuracy")

    assert response.status_code == 200
    assert response.json() == fake_report


def test_list_predictions_reflects_stored_rows(monkeypatch, tmp_path):
    test_sessionmaker = _use_isolated_db(monkeypatch, tmp_path)

    with TestClient(main.app) as client:
        with test_sessionmaker() as session:
            build_prediction_repository(session).upsert_prediction(**_prediction_kwargs())

        response = client.get("/admin/predictions")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["game_id"] == 1
    assert body[0]["predicted_margin"] == 3.5
