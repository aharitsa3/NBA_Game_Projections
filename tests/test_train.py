import numpy as np
import pandas as pd
import pytest

from nba_game_projections.models import train as train_module


def _long_row(game_id, date, team_id, opponent_team_id, is_home, **overrides):
    row = {
        "game_id": game_id,
        "date": pd.Timestamp(date),
        "season": 2021,
        "team_id": team_id,
        "opponent_team_id": opponent_team_id,
        "is_home": is_home,
        "rest_days": 2.0,
        "is_back_to_back": False,
        "elo_rating": 1500.0,
        "rolling_points": 100.0,
        "rolling_rebounds": 40.0,
        "rolling_assists": 25.0,
        "rolling_steals": 7.0,
        "rolling_blocks": 5.0,
        "rolling_turnovers": 13.0,
        "rolling_efg_pct": 0.52,
        "rolling_minutes_per_game": 240.0,
        "rolling_roster_size": 10,
    }
    row.update(overrides)
    return row


_GAME_SPECS = [
    (1, "2021-01-01", 1, 2, 100, 90),
    (2, "2021-01-03", 2, 1, 95, 105),
    (3, "2021-01-05", 1, 2, 110, 100),
]


def _long_matrix_fixture() -> pd.DataFrame:
    rows = []
    for game_id, date, home_id, away_id, _home_score, _away_score in _GAME_SPECS:
        rows.append(
            _long_row(game_id, date, home_id, away_id, True, rolling_points=90.0 + game_id)
        )
        rows.append(
            _long_row(game_id, date, away_id, home_id, False, rolling_points=80.0 + game_id)
        )
    return pd.DataFrame(rows)


def _schedules_fixture() -> pd.DataFrame:
    rows = [
        {"game_id": game_id, "home_score": home_score, "away_score": away_score}
        for game_id, _date, _home_id, _away_id, home_score, away_score in _GAME_SPECS
    ]
    return pd.DataFrame(rows)


def _patch(monkeypatch, long_df, schedules_df):
    monkeypatch.setattr(train_module, "build_feature_matrix", lambda *a, **k: long_df)
    monkeypatch.setattr(train_module, "load_schedules", lambda seasons=None: schedules_df)


def test_wide_game_matrix_pivots_home_away_correctly():
    wide = train_module.wide_game_matrix(_long_matrix_fixture())

    assert len(wide) == 3
    row = wide.loc[wide["game_id"] == 1].iloc[0]
    assert row["home_team_id"] == 1
    assert row["away_team_id"] == 2
    assert row["home_rolling_points"] == 91.0
    assert row["away_rolling_points"] == 81.0


def test_assemble_training_frame_computes_margin(monkeypatch):
    _patch(monkeypatch, _long_matrix_fixture(), _schedules_fixture())

    frame = train_module.assemble_training_frame(through_date="2021-01-10")

    row = frame.loc[frame["game_id"] == 2].iloc[0]
    assert row["margin"] == 95 - 105


def test_assemble_training_frame_raises_on_leakage(monkeypatch):
    # A poison row dated ON through_date - build_feature_matrix's real contract
    # would never produce this (covered by T2.5's own leakage tests), but this
    # proves train.py's own defensive assert independently catches it too.
    poison_date = "2021-01-10"
    long_df = pd.concat(
        [
            _long_matrix_fixture(),
            pd.DataFrame(
                [
                    _long_row(99, poison_date, 1, 2, True),
                    _long_row(99, poison_date, 2, 1, False),
                ]
            ),
        ],
        ignore_index=True,
    )
    schedules_df = pd.concat(
        [_schedules_fixture(), pd.DataFrame([{"game_id": 99, "home_score": 50, "away_score": 40}])],
        ignore_index=True,
    )
    _patch(monkeypatch, long_df, schedules_df)

    with pytest.raises(AssertionError):
        train_module.assemble_training_frame(through_date=poison_date)


def test_train_returns_fitted_model_with_working_predict(monkeypatch):
    _patch(monkeypatch, _long_matrix_fixture(), _schedules_fixture())

    model = train_module.train(
        through_date="2021-01-10",
        model_params={"n_estimators": 5, "min_child_samples": 1},
    )

    assert isinstance(model, train_module.TrainedModel)
    assert model.feature_columns == train_module.FEATURE_COLUMNS
    assert len(model.residuals) == 3
    assert np.all(np.isfinite(model.residuals))

    X = pd.DataFrame([{c: 0.0 for c in train_module.FEATURE_COLUMNS}])
    preds = model.predict(X)
    assert len(preds) == 1
    assert np.isfinite(preds[0])
