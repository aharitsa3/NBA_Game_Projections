import lightgbm as lgb
import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from nba_game_projections.app import db, main
from nba_game_projections.app.repositories.model_versions import (
    build_model_version_repository,
)


def _tiny_trained_model(through_date: str):
    from nba_game_projections.models.train import TrainedModel

    X = pd.DataFrame({"home_elo_rating": [1500.0, 1520.0]})
    y = pd.Series([5.0, -3.0])
    model = lgb.LGBMRegressor(n_estimators=2, verbosity=-1)
    model.fit(X, y)
    return TrainedModel(
        model=model,
        feature_columns=("home_elo_rating",),
        residuals=(y - model.predict(X)).to_numpy(),
        through_date=pd.Timestamp(through_date),
    )


def _use_isolated_db(monkeypatch, tmp_path):
    test_engine = create_engine(f"sqlite:///{tmp_path / 'app.db'}")
    test_sessionmaker = sessionmaker(bind=test_engine)
    monkeypatch.setattr(db, "engine", test_engine)
    monkeypatch.setattr(db, "SessionLocal", test_sessionmaker)
    monkeypatch.setattr(db.config, "MODEL_ARTIFACTS_DIR", tmp_path / "models")
    return test_sessionmaker


def test_list_models_reflects_saved_versions(monkeypatch, tmp_path):
    test_sessionmaker = _use_isolated_db(monkeypatch, tmp_path)

    with TestClient(main.app) as client:
        with test_sessionmaker() as session:
            repository = build_model_version_repository(session)
            repository.save(_tiny_trained_model("2026-01-01"))
            repository.save(_tiny_trained_model("2026-02-01"))

        response = client.get("/admin/models")

    assert response.status_code == 200
    versions = response.json()
    assert len(versions) == 2
    assert all(v["is_active"] is False for v in versions)


def test_activate_flips_active_flag_and_deactivates_previous(monkeypatch, tmp_path):
    test_sessionmaker = _use_isolated_db(monkeypatch, tmp_path)

    with TestClient(main.app) as client:
        with test_sessionmaker() as session:
            repository = build_model_version_repository(session)
            first_id = repository.save(_tiny_trained_model("2026-01-01")).id
            second_id = repository.save(_tiny_trained_model("2026-02-01")).id
            repository.activate(first_id)

        activate_response = client.post(f"/admin/models/{second_id}/activate")
        assert activate_response.status_code == 200
        assert activate_response.json()["is_active"] is True

        list_response = client.get("/admin/models")
        versions = {v["id"]: v["is_active"] for v in list_response.json()}
        assert versions[first_id] is False
        assert versions[second_id] is True


def test_activate_unknown_version_returns_404(monkeypatch, tmp_path):
    _use_isolated_db(monkeypatch, tmp_path)

    with TestClient(main.app) as client:
        response = client.post("/admin/models/999/activate")

    assert response.status_code == 404
