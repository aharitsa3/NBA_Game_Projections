import lightgbm as lgb
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from nba_game_projections import config
from nba_game_projections.app import db
from nba_game_projections.app.repositories.model_versions import build_model_version_repository
from nba_game_projections.app.services import training
from nba_game_projections.models.train import TrainedModel


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_ARTIFACTS_DIR", tmp_path / "models")
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    db.Base.metadata.create_all(bind=engine)
    with sessionmaker(bind=engine)() as db_session:
        yield db_session


def _tiny_trained_model() -> TrainedModel:
    X = pd.DataFrame({"home_elo_rating": [1500.0, 1520.0]})
    y = pd.Series([5.0, -3.0])
    model = lgb.LGBMRegressor(n_estimators=2, verbosity=-1)
    model.fit(X, y)
    return TrainedModel(
        model=model,
        feature_columns=("home_elo_rating",),
        residuals=(y - model.predict(X)).to_numpy(),
        through_date=pd.Timestamp("2026-06-01"),
    )


def test_run_production_training_saves_and_activates_model(session, monkeypatch):
    monkeypatch.setattr(training, "train", lambda through_date, seasons=None: _tiny_trained_model())
    monkeypatch.setattr(training, "capture_pending_predictions", lambda s: [])

    activated = training.run_production_training(session)

    assert activated.is_active is True
    assert build_model_version_repository(session).get_active().id == activated.id


def test_run_production_training_refreshes_pending_predictions(session, monkeypatch):
    monkeypatch.setattr(training, "train", lambda through_date, seasons=None: _tiny_trained_model())
    captured_with = []
    monkeypatch.setattr(
        training, "capture_pending_predictions", lambda s: captured_with.append(s) or []
    )

    training.run_production_training(session)

    assert captured_with == [session]
