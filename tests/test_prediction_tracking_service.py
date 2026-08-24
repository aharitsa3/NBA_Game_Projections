import lightgbm as lgb
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from nba_game_projections import config
from nba_game_projections.app import db
from nba_game_projections.app.repositories.model_versions import build_model_version_repository
from nba_game_projections.app.repositories.predictions import build_prediction_repository
from nba_game_projections.app.services import prediction_tracking
from nba_game_projections.app.services.prediction import NoActiveModelError
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


def _activate_model(session) -> int:
    repository = build_model_version_repository(session)
    record = repository.save(_tiny_trained_model())
    return repository.activate(record.id).id


def _prediction_dict(**overrides) -> dict:
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
    }
    fields.update(overrides)
    return fields


def test_capture_upserts_predictions_with_active_model_id(session, monkeypatch):
    active_id = _activate_model(session)
    monkeypatch.setattr(
        prediction_tracking,
        "predict_pending_games",
        lambda s, horizon_days=14: [_prediction_dict()],
    )

    records = prediction_tracking.capture_pending_predictions(session)

    assert len(records) == 1
    assert records[0].game_id == 1
    assert records[0].model_version_id == active_id


def test_capture_refreshes_pending_row_on_second_call(session, monkeypatch):
    _activate_model(session)
    monkeypatch.setattr(
        prediction_tracking,
        "predict_pending_games",
        lambda s, horizon_days=14: [_prediction_dict(predicted_margin=3.5)],
    )
    prediction_tracking.capture_pending_predictions(session)

    monkeypatch.setattr(
        prediction_tracking,
        "predict_pending_games",
        lambda s, horizon_days=14: [_prediction_dict(predicted_margin=9.0)],
    )
    records = prediction_tracking.capture_pending_predictions(session)

    assert records[0].predicted_margin == 9.0


def test_capture_never_overwrites_reconciled_row(session, monkeypatch):
    _activate_model(session)
    monkeypatch.setattr(
        prediction_tracking,
        "predict_pending_games",
        lambda s, horizon_days=14: [_prediction_dict(predicted_margin=3.5)],
    )
    prediction_tracking.capture_pending_predictions(session)
    build_prediction_repository(session).reconcile(1, actual_margin=5.0, actual_winner="home")

    monkeypatch.setattr(
        prediction_tracking,
        "predict_pending_games",
        lambda s, horizon_days=14: [_prediction_dict(predicted_margin=99.0)],
    )
    records = prediction_tracking.capture_pending_predictions(session)

    assert records[0].predicted_margin == 3.5


def test_capture_propagates_no_active_model_error(session, monkeypatch):
    def raise_no_active_model(s, horizon_days=14):
        raise NoActiveModelError("no active model")

    monkeypatch.setattr(prediction_tracking, "predict_pending_games", raise_no_active_model)

    with pytest.raises(NoActiveModelError):
        prediction_tracking.capture_pending_predictions(session)


def test_capture_returns_empty_list_when_no_pending_games(session, monkeypatch):
    _activate_model(session)
    monkeypatch.setattr(prediction_tracking, "predict_pending_games", lambda s, horizon_days=14: [])

    assert prediction_tracking.capture_pending_predictions(session) == []


def _completed_schedules_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [1, 2],
            "home_score": [100, 95],
            "away_score": [90, 98],
            "home_winner": [True, False],
        }
    )


def test_reconcile_fills_actual_result_for_completed_game(session, monkeypatch):
    _activate_model(session)
    repository = build_prediction_repository(session)
    repository.upsert_prediction(model_version_id=1, **_prediction_dict(game_id=1))
    monkeypatch.setattr(
        prediction_tracking, "load_schedules", lambda seasons=None: _completed_schedules_fixture()
    )

    reconciled = prediction_tracking.reconcile_predictions(session)

    assert len(reconciled) == 1
    assert reconciled[0].actual_margin == 10.0
    assert reconciled[0].actual_winner == "home"
    assert reconciled[0].reconciled_at is not None


def test_reconcile_skips_games_not_yet_complete_locally(session, monkeypatch):
    _activate_model(session)
    repository = build_prediction_repository(session)
    repository.upsert_prediction(model_version_id=1, **_prediction_dict(game_id=404))
    monkeypatch.setattr(
        prediction_tracking, "load_schedules", lambda seasons=None: _completed_schedules_fixture()
    )

    assert prediction_tracking.reconcile_predictions(session) == []
    assert repository.get_by_game_id(404).reconciled_at is None


def test_reconcile_returns_empty_list_when_nothing_pending(session, monkeypatch):
    monkeypatch.setattr(
        prediction_tracking, "load_schedules", lambda seasons=None: _completed_schedules_fixture()
    )

    assert prediction_tracking.reconcile_predictions(session) == []


def test_compute_accuracy_returns_none_pooled_when_nothing_reconciled(session):
    assert prediction_tracking.compute_accuracy(session) == {"pooled": None, "by_model_version": []}


def test_compute_accuracy_summarizes_reconciled_predictions(session):
    repository = build_prediction_repository(session)
    repository.upsert_prediction(
        model_version_id=1,
        **_prediction_dict(
            game_id=1, predicted_margin=5.0, predicted_winner="home", home_win_probability=0.8
        ),
    )
    repository.reconcile(1, actual_margin=10.0, actual_winner="home")

    repository.upsert_prediction(
        model_version_id=2,
        **_prediction_dict(
            game_id=2, predicted_margin=-2.0, predicted_winner="away", home_win_probability=0.3
        ),
    )
    repository.reconcile(2, actual_margin=4.0, actual_winner="home")

    report = prediction_tracking.compute_accuracy(session)

    assert report["pooled"]["n_games"] == 2
    assert report["pooled"]["accuracy"] == 0.5
    versions = {row["model_version_id"]: row for row in report["by_model_version"]}
    assert versions[1]["accuracy"] == 1.0
    assert versions[2]["accuracy"] == 0.0
