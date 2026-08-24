import numpy as np
import pandas as pd

from nba_game_projections.features import context

T1, T2, T3 = 1610612737, 1610612738, 1610612739


def _schedule_row(game_id, date, season, home_team_id, away_team_id):
    return {
        "game_id": game_id,
        "season": season,
        "season_type": 2,
        "date": pd.Timestamp(date, tz="America/New_York"),
        "home_team_id": home_team_id,
        "home_team_name": f"Team {home_team_id}",
        "home_team_abbreviation": f"H{home_team_id}",
        "home_score": 100,
        "away_team_id": away_team_id,
        "away_team_name": f"Team {away_team_id}",
        "away_team_abbreviation": f"A{away_team_id}",
        "away_score": 95,
        "home_winner": True,
        "away_winner": False,
    }


def _schedules_df() -> pd.DataFrame:
    rows = [
        # T1 home vs T2 away, T1's and T2's first game -> both rest_days NaN.
        _schedule_row(1, "2020-01-01 19:30:00", 2020, T1, T2),
        # T1 away vs T3, one day later -> back-to-back for T1; T3's first
        # game -> rest_days NaN for T3.
        _schedule_row(2, "2020-01-02 19:30:00", 2020, T3, T1),
        # T1 home vs T2, 4-day gap after game 2 -> bye for T1 (rest_days=4).
        # For T2, 5-day gap after game 1 -> rest_days=5, not a back-to-back.
        _schedule_row(3, "2020-01-06 19:30:00", 2020, T1, T2),
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_output_columns_and_row_count():
    result = context.build_team_game_context(_schedules_df())

    assert list(result.columns) == [
        "game_id",
        "date",
        "season",
        "team_id",
        "opponent_team_id",
        "is_home",
        "rest_days",
        "is_back_to_back",
    ]
    assert len(result) == 6


def test_home_and_away_rows_for_a_game():
    result = context.build_team_game_context(_schedules_df())
    game_1 = result[result["game_id"] == 1]

    home_row = game_1[game_1["team_id"] == T1].iloc[0]
    assert home_row["is_home"]
    assert home_row["opponent_team_id"] == T2

    away_row = game_1[game_1["team_id"] == T2].iloc[0]
    assert not away_row["is_home"]
    assert away_row["opponent_team_id"] == T1


def test_first_game_has_nan_rest_and_no_back_to_back():
    result = context.build_team_game_context(_schedules_df())
    first_games = result[result["game_id"] == 1]

    for _, row in first_games.iterrows():
        assert np.isnan(row["rest_days"])
        assert not row["is_back_to_back"]

    t3_first_game = result[(result["team_id"] == T3) & (result["game_id"] == 2)].iloc[0]
    assert np.isnan(t3_first_game["rest_days"])
    assert not t3_first_game["is_back_to_back"]


def test_back_to_back_flagged_correctly():
    result = context.build_team_game_context(_schedules_df())
    t1_game_2 = result[(result["team_id"] == T1) & (result["game_id"] == 2)].iloc[0]

    assert t1_game_2["rest_days"] == 1
    assert t1_game_2["is_back_to_back"]


def test_bye_gap_flagged_correctly():
    result = context.build_team_game_context(_schedules_df())

    t1_game_3 = result[(result["team_id"] == T1) & (result["game_id"] == 3)].iloc[0]
    assert t1_game_3["rest_days"] == 4
    assert not t1_game_3["is_back_to_back"]

    t2_game_3 = result[(result["team_id"] == T2) & (result["game_id"] == 3)].iloc[0]
    assert t2_game_3["rest_days"] == 5
    assert not t2_game_3["is_back_to_back"]
