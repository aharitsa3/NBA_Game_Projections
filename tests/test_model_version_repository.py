import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from nba_game_projections import config
from nba_game_projections.app.db import Base
from nba_game_projections.app.repositories.model_versions import (
    build_model_version_repository,
)
from nba_game_projections.models.train import TrainedModel


def _tiny_trained_model(through_date: str) -> TrainedModel:
    X = pd.DataFrame({"home_elo_rating": [1500.0, 1520.0, 1480.0, 1510.0]})
    y = pd.Series([5.0, 3.0, -2.0, 4.0])
    model = lgb.LGBMRegressor(n_estimators=2, verbosity=-1)
    model.fit(X, y)
    residuals = (y - model.predict(X)).to_numpy()
    return TrainedModel(
        model=model,
        feature_columns=("home_elo_rating",),
        residuals=residuals,
        through_date=pd.Timestamp(through_date),
    )


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(bind=engine)
    with sessionmaker(bind=engine)() as db_session:
        yield db_session


@pytest.fixture
def repo(session, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_ARTIFACTS_DIR", tmp_path / "models")
    return build_model_version_repository(session)


def test_save_writes_artifact_and_metadata(repo):
    record = repo.save(_tiny_trained_model("2026-01-01"))

    assert record.id is not None
    assert record.is_active is False
    from pathlib import Path

    assert Path(record.artifact_path).exists()


def test_activate_second_version_deactivates_first(repo):
    first = repo.save(_tiny_trained_model("2026-01-01"))
    second = repo.save(_tiny_trained_model("2026-02-01"))

    repo.activate(second.id)

    versions = {v.id: v.is_active for v in repo.list()}
    assert versions[first.id] is False
    assert versions[second.id] is True
    assert repo.get_active().id == second.id


def test_activate_missing_id_returns_none(repo):
    assert repo.activate(999) is None


def test_load_active_model_round_trips(repo):
    trained = _tiny_trained_model("2026-01-01")
    record = repo.save(trained)
    repo.activate(record.id)

    loaded = repo.load_active_model()

    assert loaded.feature_columns == trained.feature_columns
    np.testing.assert_array_equal(loaded.residuals, trained.residuals)
    assert loaded.through_date == trained.through_date
    predictions = loaded.predict(pd.DataFrame({"home_elo_rating": [1500.0]}))
    assert predictions.shape == (1,)


def test_load_active_model_returns_none_when_no_active_version(repo):
    assert repo.load_active_model() is None
