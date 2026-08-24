import pandas as pd
import pytest

from nba_game_projections.features import elo


def _schedule_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_two_game_worked_example():
    # Team A (id 1) hosts Team B (id 2) and wins, then A hosts B again.
    schedule = _schedule_df(
        [
            {
                "game_id": 1,
                "date": pd.Timestamp("2020-01-01"),
                "home_team_id": 1,
                "away_team_id": 2,
                "home_score": 110,
                "away_score": 100,
            },
            {
                "game_id": 2,
                "date": pd.Timestamp("2020-01-03"),
                "home_team_id": 1,
                "away_team_id": 2,
                "home_score": 95,
                "away_score": 105,
            },
        ]
    )

    result = elo.compute_elo_ratings(
        schedule, k_factor=20.0, initial_rating=1500.0, home_advantage=100.0
    )

    game1 = result[result["game_id"] == 1]
    a_pre_g1 = game1[game1["team_id"] == 1]["pre_game_rating"].iloc[0]
    b_pre_g1 = game1[game1["team_id"] == 2]["pre_game_rating"].iloc[0]
    assert a_pre_g1 == pytest.approx(1500.0, abs=1e-9)
    assert b_pre_g1 == pytest.approx(1500.0, abs=1e-9)

    game2 = result[result["game_id"] == 2]
    a_pre_g2 = game2[game2["team_id"] == 1]["pre_game_rating"].iloc[0]
    b_pre_g2 = game2[game2["team_id"] == 2]["pre_game_rating"].iloc[0]
    assert a_pre_g2 == pytest.approx(1507.1987, abs=1e-3)
    assert b_pre_g2 == pytest.approx(1492.8013, abs=1e-3)


def test_new_team_starts_at_initial_rating():
    schedule = _schedule_df(
        [
            {
                "game_id": 1,
                "date": pd.Timestamp("2020-01-01"),
                "home_team_id": 1,
                "away_team_id": 2,
                "home_score": 110,
                "away_score": 100,
            },
            {
                "game_id": 2,
                "date": pd.Timestamp("2020-01-03"),
                "home_team_id": 1,
                "away_team_id": 2,
                "home_score": 100,
                "away_score": 90,
            },
            {
                "game_id": 3,
                "date": pd.Timestamp("2020-01-05"),
                "home_team_id": 3,
                "away_team_id": 1,
                "home_score": 90,
                "away_score": 100,
            },
        ]
    )

    result = elo.compute_elo_ratings(schedule, initial_rating=1500.0)

    game3 = result[result["game_id"] == 3]
    team3_pre = game3[game3["team_id"] == 3]["pre_game_rating"].iloc[0]
    assert team3_pre == pytest.approx(1500.0, abs=1e-9)

    team1_pre_g3 = game3[game3["team_id"] == 1]["pre_game_rating"].iloc[0]
    assert team1_pre_g3 != pytest.approx(1500.0, abs=1e-9)


def test_output_columns():
    schedule = _schedule_df(
        [
            {
                "game_id": 1,
                "date": pd.Timestamp("2020-01-01"),
                "home_team_id": 1,
                "away_team_id": 2,
                "home_score": 110,
                "away_score": 100,
            }
        ]
    )
    result = elo.compute_elo_ratings(schedule)
    assert list(result.columns) == [
        "game_id",
        "date",
        "team_id",
        "opponent_team_id",
        "is_home",
        "pre_game_rating",
    ]
    assert len(result) == 2
    home_row = result[result["is_home"]].iloc[0]
    away_row = result[~result["is_home"]].iloc[0]
    assert home_row["team_id"] == 1
    assert home_row["opponent_team_id"] == 2
    assert away_row["team_id"] == 2
    assert away_row["opponent_team_id"] == 1


def test_win_probability_equal_ratings_zero_home_advantage():
    assert elo.elo_win_probability(1500.0, 1500.0, home_advantage=0.0) == pytest.approx(0.5)


def test_win_probability_monotonic_in_rating_gap():
    low_gap = elo.elo_win_probability(1500.0, 1490.0, home_advantage=0.0)
    mid_gap = elo.elo_win_probability(1500.0, 1400.0, home_advantage=0.0)
    high_gap = elo.elo_win_probability(1500.0, 1200.0, home_advantage=0.0)
    assert low_gap < mid_gap < high_gap

    more_home_advantage = elo.elo_win_probability(1500.0, 1500.0, home_advantage=50.0)
    less_home_advantage = elo.elo_win_probability(1500.0, 1500.0, home_advantage=10.0)
    assert less_home_advantage < more_home_advantage


def test_k_factor_changes_output():
    schedule = _schedule_df(
        [
            {
                "game_id": 1,
                "date": pd.Timestamp("2020-01-01"),
                "home_team_id": 1,
                "away_team_id": 2,
                "home_score": 110,
                "away_score": 100,
            },
            {
                "game_id": 2,
                "date": pd.Timestamp("2020-01-03"),
                "home_team_id": 1,
                "away_team_id": 2,
                "home_score": 90,
                "away_score": 100,
            },
        ]
    )

    default_result = elo.compute_elo_ratings(schedule, k_factor=20.0)
    high_k_result = elo.compute_elo_ratings(schedule, k_factor=40.0)

    default_g2 = default_result[
        (default_result["game_id"] == 2) & (default_result["team_id"] == 1)
    ]["pre_game_rating"].iloc[0]
    high_k_g2 = high_k_result[(high_k_result["game_id"] == 2) & (high_k_result["team_id"] == 1)][
        "pre_game_rating"
    ].iloc[0]
    assert default_g2 != pytest.approx(high_k_g2)


def test_initial_rating_changes_output():
    schedule = _schedule_df(
        [
            {
                "game_id": 1,
                "date": pd.Timestamp("2020-01-01"),
                "home_team_id": 1,
                "away_team_id": 2,
                "home_score": 110,
                "away_score": 100,
            }
        ]
    )

    default_result = elo.compute_elo_ratings(schedule, initial_rating=1500.0)
    other_result = elo.compute_elo_ratings(schedule, initial_rating=1600.0)

    default_pre = default_result[default_result["team_id"] == 1]["pre_game_rating"].iloc[0]
    other_pre = other_result[other_result["team_id"] == 1]["pre_game_rating"].iloc[0]
    assert default_pre == pytest.approx(1500.0)
    assert other_pre == pytest.approx(1600.0)
    assert default_pre != pytest.approx(other_pre)


def test_home_advantage_changes_output():
    schedule = _schedule_df(
        [
            {
                "game_id": 1,
                "date": pd.Timestamp("2020-01-01"),
                "home_team_id": 1,
                "away_team_id": 2,
                "home_score": 110,
                "away_score": 100,
            },
            {
                "game_id": 2,
                "date": pd.Timestamp("2020-01-03"),
                "home_team_id": 1,
                "away_team_id": 2,
                "home_score": 90,
                "away_score": 100,
            },
        ]
    )

    low_ha_result = elo.compute_elo_ratings(schedule, home_advantage=0.0)
    high_ha_result = elo.compute_elo_ratings(schedule, home_advantage=200.0)

    low_ha_g2 = low_ha_result[(low_ha_result["game_id"] == 2) & (low_ha_result["team_id"] == 1)][
        "pre_game_rating"
    ].iloc[0]
    high_ha_g2 = high_ha_result[
        (high_ha_result["game_id"] == 2) & (high_ha_result["team_id"] == 1)
    ]["pre_game_rating"].iloc[0]
    assert low_ha_g2 != pytest.approx(high_ha_g2)
