import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from nba_game_projections import config
from nba_game_projections.app import db
from nba_game_projections.app.repositories.model_versions import build_model_version_repository
from nba_game_projections.app.repositories.roster_overrides import (
    build_roster_override_repository,
)
from nba_game_projections.app.services import prediction
from nba_game_projections.app.services.prediction import NoActiveModelError
from nba_game_projections.features import build as build_module
from nba_game_projections.models.train import FEATURE_COLUMNS, TrainedModel

_TARGET_DATE = "2027-01-15"
_FUTURE_SEASON = 2027


def _completed_schedules_fixture() -> pd.DataFrame:
    """Real completed history (`prediction.load_schedules`'s clean-schema output)."""
    return pd.DataFrame(
        {
            "game_id": [1, 2],
            "season": [2020, 2020],
            "season_type": [2, 2],
            "date": pd.to_datetime(["2020-11-01 19:30", "2020-11-05 19:30"], utc=True),
            "home_team_id": [1, 2],
            "home_team_name": ["Team One", "Team Two"],
            "home_team_abbreviation": ["T1", "T2"],
            "home_score": [100, 95],
            "away_team_id": [2, 1],
            "away_team_name": ["Team Two", "Team One"],
            "away_team_abbreviation": ["T2", "T1"],
            "away_score": [90, 105],
            "home_winner": [True, False],
            "away_winner": [False, True],
        }
    )


def _upcoming_games_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [100],
            "season": [_FUTURE_SEASON],
            "season_type": [2],
            "date": pd.to_datetime([f"{_TARGET_DATE} 19:30"], utc=True),
            "home_team_id": [1],
            "home_team_name": ["Team One"],
            "away_team_id": [2],
            "away_team_name": ["Team Two"],
        }
    )


def _near_term_upcoming_games_fixture(days_ahead: int = 5) -> pd.DataFrame:
    """A game within `predict_pending_games`'s 14-day horizon, dated relative to real "now"."""
    game_date = pd.Timestamp.now().normalize() + pd.Timedelta(days=days_ahead)
    return pd.DataFrame(
        {
            "game_id": [200],
            "season": [_FUTURE_SEASON],
            "season_type": [2],
            "date": pd.to_datetime([game_date.strftime("%Y-%m-%d 19:30")], utc=True),
            "home_team_id": [1],
            "home_team_name": ["Team One"],
            "away_team_id": [2],
            "away_team_name": ["Team Two"],
        }
    )


def _fake_load_upcoming_games_near_term(season=None):
    """Like the autouse fixture's loader, but the one non-empty season returns a near-term game.

    Scoping the near-term game to a single season label matters: `_all_upcoming_games()`
    unions `current_season()` and `current_season() + 1`, so a loader that returned the
    same game_id for *every* season would create duplicate rows for that game_id.
    """
    if season == _FUTURE_SEASON:
        return _near_term_upcoming_games_fixture()
    return _empty_upcoming_games_fixture()


def _empty_upcoming_games_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "game_id",
            "season",
            "season_type",
            "date",
            "home_team_id",
            "home_team_name",
            "away_team_id",
            "away_team_name",
        ]
    )


def _player_row(**overrides) -> dict:
    row = {
        "game_id": 1,
        "season": 2020,
        "game_date": pd.Timestamp("2020-11-01"),
        "athlete_id": 11,
        "athlete_display_name": "Player Eleven",
        "team_id": 1,
        "home_away": "home",
        "opponent_team_id": 2,
        "starter": True,
        "did_not_play": False,
        "dnp_reason": None,
        "minutes": 30.0,
        "points": 20.0,
        "rebounds": 5.0,
        "offensive_rebounds": 1.0,
        "defensive_rebounds": 4.0,
        "assists": 3.0,
        "steals": 1.0,
        "blocks": 0.0,
        "turnovers": 2.0,
        "fouls": 2.0,
        "field_goals_made": 8.0,
        "field_goals_attempted": 15.0,
        "three_point_field_goals_made": 2.0,
        "three_point_field_goals_attempted": 5.0,
        "free_throws_made": 2.0,
        "free_throws_attempted": 2.0,
    }
    row.update(overrides)
    return row


def _player_box_fixture() -> pd.DataFrame:
    """One player per team, identical stat line across both completed games -
    team 1's lone player (11) drives `home_rolling_points`, the feature the
    stub model below is sensitive to."""
    rows = [
        _player_row(game_id=1, game_date=pd.Timestamp("2020-11-01"), athlete_id=11, team_id=1),
        _player_row(game_id=2, game_date=pd.Timestamp("2020-11-05"), athlete_id=11, team_id=1),
        _player_row(
            game_id=1,
            game_date=pd.Timestamp("2020-11-01"),
            athlete_id=21,
            team_id=2,
            points=15.0,
        ),
        _player_row(
            game_id=2,
            game_date=pd.Timestamp("2020-11-05"),
            athlete_id=21,
            team_id=2,
            points=15.0,
        ),
    ]
    return pd.DataFrame(rows)


def _stub_trained_model() -> TrainedModel:
    """A model fit to depend only on `home_rolling_points` (margin ~ home_rolling_points),
    every other column held constant - so removing team 1's only player (dropping
    `home_rolling_points` from 20 to 0) reliably shifts the prediction, while every
    other real, uncontrolled feature value (elo, rest days, ...) has zero learned effect.
    """
    n = 11
    home_points = np.linspace(0.0, 30.0, n)
    X = pd.DataFrame({col: 0.0 for col in FEATURE_COLUMNS}, index=range(n))
    X["home_rolling_points"] = home_points
    y = pd.Series(home_points)

    model = lgb.LGBMRegressor(
        n_estimators=30, min_child_samples=1, min_data_in_leaf=1, verbosity=-1
    )
    model.fit(X, y)
    residuals = (y - model.predict(X)).to_numpy()
    return TrainedModel(
        model=model,
        feature_columns=FEATURE_COLUMNS,
        residuals=residuals,
        through_date=pd.Timestamp("2026-06-01"),
    )


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_ARTIFACTS_DIR", tmp_path / "models")
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    db.Base.metadata.create_all(bind=engine)
    with sessionmaker(bind=engine)() as db_session:
        yield db_session


@pytest.fixture(autouse=True)
def _patch_loaders(monkeypatch):
    monkeypatch.setattr(
        prediction, "load_schedules", lambda seasons=None: _completed_schedules_fixture()
    )

    def fake_load_upcoming_games(season=None):
        if season == _FUTURE_SEASON:
            return _upcoming_games_fixture()
        return _empty_upcoming_games_fixture()

    monkeypatch.setattr(prediction, "load_upcoming_games", fake_load_upcoming_games)
    monkeypatch.setattr(build_module, "load_player_box", lambda seasons=None: _player_box_fixture())
    monkeypatch.setattr(config, "current_season", lambda: _FUTURE_SEASON - 1)


def _activate_stub_model(session) -> None:
    repository = build_model_version_repository(session)
    record = repository.save(_stub_trained_model())
    repository.activate(record.id)


def test_predict_slate_returns_expected_shape(session):
    _activate_stub_model(session)

    predictions = prediction.predict_slate(session, date=_TARGET_DATE)

    assert len(predictions) == 1
    result = predictions[0]
    assert result["game_id"] == 100
    assert result["home_team_id"] == 1
    assert result["home_team_name"] == "Team One"
    assert result["away_team_id"] == 2
    assert result["away_team_name"] == "Team Two"
    assert isinstance(result["predicted_margin"], float)
    assert result["predicted_winner"] in {"home", "away"}
    assert 0.0 <= result["home_win_probability"] <= 1.0


def test_predict_slate_raises_without_active_model(session):
    with pytest.raises(NoActiveModelError):
        prediction.predict_slate(session, date=_TARGET_DATE)


def test_predict_slate_returns_empty_list_when_no_games_on_date(session):
    assert prediction.predict_slate(session, date="2030-06-01") == []


def test_predict_pending_games_returns_games_within_horizon(session, monkeypatch):
    _activate_stub_model(session)
    monkeypatch.setattr(prediction, "load_upcoming_games", _fake_load_upcoming_games_near_term)

    predictions = prediction.predict_pending_games(session)

    assert len(predictions) == 1
    assert predictions[0]["game_id"] == 200


def test_predict_pending_games_honors_horizon_days_override(session, monkeypatch):
    """`_near_term_upcoming_games_fixture` is dated 5 days out - a 1-day horizon excludes it."""
    _activate_stub_model(session)
    monkeypatch.setattr(prediction, "load_upcoming_games", _fake_load_upcoming_games_near_term)

    assert prediction.predict_pending_games(session, horizon_days=1) == []


def test_predict_pending_games_excludes_games_beyond_horizon(session):
    """The autouse `_patch_loaders` fixture's game is dated `_TARGET_DATE` (2027-01-15) -
    far beyond the 14-day pending horizon from real "now"."""
    _activate_stub_model(session)

    assert prediction.predict_pending_games(session) == []


def test_predict_pending_games_raises_without_active_model(session, monkeypatch):
    monkeypatch.setattr(prediction, "load_upcoming_games", _fake_load_upcoming_games_near_term)

    with pytest.raises(NoActiveModelError):
        prediction.predict_pending_games(session)


def test_predict_pending_games_excludes_completed_games_inside_the_widened_window(
    session, monkeypatch
):
    """Regression test: a real completed game whose date falls inside the feature-matrix
    window (widened to span every target game) must never leak into the output - it has
    no placeholder row in `target_games` to key a prediction dict off of."""
    _activate_stub_model(session)
    monkeypatch.setattr(prediction, "load_upcoming_games", _fake_load_upcoming_games_near_term)
    stray_completed_date = pd.Timestamp.now().normalize() + pd.Timedelta(days=2)
    stray_completed_game = pd.DataFrame(
        {
            "game_id": [999],
            "season": [_FUTURE_SEASON],
            "season_type": [2],
            "date": pd.to_datetime([stray_completed_date], utc=True),
            "home_team_id": [1],
            "home_team_name": ["Team One"],
            "home_team_abbreviation": ["T1"],
            "home_score": [110],
            "away_team_id": [2],
            "away_team_name": ["Team Two"],
            "away_team_abbreviation": ["T2"],
            "away_score": [100],
            "home_winner": [True],
            "away_winner": [False],
        }
    )
    monkeypatch.setattr(
        prediction,
        "load_schedules",
        lambda seasons=None: pd.concat(
            [_completed_schedules_fixture(), stray_completed_game], ignore_index=True
        ),
    )

    predictions = prediction.predict_pending_games(session)

    assert [p["game_id"] for p in predictions] == [200]


def test_roster_override_change_alters_next_calls_prediction(session):
    """The core no-caching guarantee (PRD §17): a roster-override edit changes
    the very next call's numbers, with no separate recompute step."""
    _activate_stub_model(session)

    baseline = prediction.predict_slate(session, date=_TARGET_DATE)[0]

    build_roster_override_repository(session).add(
        player_id=11, player_name="Player Eleven", team_id=1, status="out"
    )

    after_override = prediction.predict_slate(session, date=_TARGET_DATE)[0]

    assert after_override["predicted_margin"] != baseline["predicted_margin"]
    assert after_override["home_win_probability"] != baseline["home_win_probability"]
