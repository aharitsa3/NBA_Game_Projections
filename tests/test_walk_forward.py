import numpy as np
import pandas as pd
import pytest

from nba_game_projections.backtest import walk_forward


class _StubModel:
    residuals = np.array([1.0, -1.0, 2.0])

    def predict(self, X):
        return np.full(len(X), 5.0)


_GAMES = [
    # (game_id, season, date, home_id, away_id, home_score, away_score)
    (101, 2017, "2017-11-01", 1, 2, 100, 90),
    (102, 2017, "2017-11-05", 2, 1, 95, 105),
    (201, 2018, "2018-11-01", 1, 2, 100, 90),
    (202, 2018, "2018-11-05", 2, 1, 95, 105),
    (301, 2019, "2019-11-01", 1, 2, 100, 90),
    (302, 2019, "2019-11-05", 2, 1, 95, 105),
]


def _long_row(game_id, date, team_id, opponent_team_id, is_home, season):
    return {
        "game_id": game_id,
        "date": pd.Timestamp(date),
        "season": season,
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


def _master_long() -> pd.DataFrame:
    rows = []
    for game_id, season, date, home_id, away_id, _hs, _as in _GAMES:
        rows.append(_long_row(game_id, date, home_id, away_id, True, season))
        rows.append(_long_row(game_id, date, away_id, home_id, False, season))
    return pd.DataFrame(rows)


def _schedules_fixture() -> pd.DataFrame:
    rows = [
        {
            "game_id": game_id,
            "season": season,
            "date": pd.Timestamp(date, tz="America/New_York"),
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_score": home_score,
            "away_score": away_score,
        }
        for game_id, season, date, home_id, away_id, home_score, away_score in _GAMES
    ]
    return pd.DataFrame(rows)


def _patch(monkeypatch, calls):
    def fake_train(through_date, seasons=None, **kwargs):
        calls.append(("train", pd.Timestamp(through_date), list(seasons)))
        return _StubModel()

    def fake_build_feature_matrix(start_date, end_date, seasons=None, **kwargs):
        calls.append(
            (
                "build_feature_matrix",
                pd.Timestamp(start_date),
                pd.Timestamp(end_date),
                list(seasons),
            )
        )
        start_date, end_date = pd.Timestamp(start_date), pd.Timestamp(end_date)
        master = _master_long()
        mask = (master["date"] >= start_date) & (master["date"] < end_date)
        return master.loc[mask].reset_index(drop=True)

    monkeypatch.setattr(walk_forward, "load_schedules", lambda seasons=None: _schedules_fixture())
    monkeypatch.setattr(walk_forward, "train", fake_train)
    monkeypatch.setattr(walk_forward, "build_feature_matrix", fake_build_feature_matrix)


def test_fold_partitioning_skips_first_season_and_expands_training_set(monkeypatch):
    calls = []
    _patch(monkeypatch, calls)

    result = walk_forward.run_walk_forward(seasons=[2017, 2018, 2019], min_train_seasons=1)

    train_calls = [c for c in calls if c[0] == "train"]
    assert len(train_calls) == 2
    assert train_calls[0][2] == [2017]
    assert train_calls[1][2] == [2017, 2018]

    bfm_calls = [c for c in calls if c[0] == "build_feature_matrix"]
    assert bfm_calls[0][3] == [2017, 2018]
    assert bfm_calls[1][3] == [2017, 2018, 2019]

    assert set(result["test_season"]) == {2018, 2019}
    assert len(result) == 4  # 2 games per test season x 2 test seasons


def test_no_leakage_test_season_2017_never_appears_as_a_fold(monkeypatch):
    calls = []
    _patch(monkeypatch, calls)

    result = walk_forward.run_walk_forward(seasons=[2017, 2018, 2019], min_train_seasons=1)

    assert 2017 not in set(result["test_season"])
    assert set(result["game_id"]) == {201, 202, 301, 302}


def test_predictions_and_actuals_computed_correctly(monkeypatch):
    calls = []
    _patch(monkeypatch, calls)

    result = walk_forward.run_walk_forward(seasons=[2017, 2018, 2019], min_train_seasons=1)

    row = result.loc[result["game_id"] == 201].iloc[0]
    assert row["predicted_margin"] == 5.0
    assert bool(row["predicted_home_win"]) is True
    assert row["actual_margin"] == 100 - 90
    assert bool(row["actual_home_win"]) is True
    assert 0.0 < row["predicted_win_probability"] < 1.0

    row = result.loc[result["game_id"] == 202].iloc[0]
    assert row["actual_margin"] == 95 - 105
    assert bool(row["actual_home_win"]) is False
    # model is a constant-5.0 stub - it still "predicts" home win here even
    # though home actually lost, proving predicted/actual are independent
    assert bool(row["predicted_home_win"]) is True


def test_raises_on_leakage_if_a_fold_pulls_in_a_future_row(monkeypatch):
    def fake_train(through_date, seasons=None, **kwargs):
        return _StubModel()

    def leaky_build_feature_matrix(start_date, end_date, seasons=None, **kwargs):
        # deliberately ignores the requested range and returns everything,
        # simulating a bug in a lower layer - this proves run_walk_forward's
        # own defensive assert independently catches it.
        return _master_long()

    monkeypatch.setattr(walk_forward, "load_schedules", lambda seasons=None: _schedules_fixture())
    monkeypatch.setattr(walk_forward, "train", fake_train)
    monkeypatch.setattr(walk_forward, "build_feature_matrix", leaky_build_feature_matrix)

    with pytest.raises(AssertionError):
        walk_forward.run_walk_forward(seasons=[2017, 2018, 2019], min_train_seasons=1)


def test_raises_when_no_fold_is_possible(monkeypatch):
    calls = []
    _patch(monkeypatch, calls)

    with pytest.raises(ValueError):
        walk_forward.run_walk_forward(seasons=[2017], min_train_seasons=1)
