import pandas as pd

from nba_game_projections.backtest import baselines

TEAM_A, TEAM_B, TEAM_C, TEAM_D = 1, 2, 3, 4


def _schedule_row(
    game_id,
    date,
    season,
    home_team_id,
    away_team_id,
    home_score,
    away_score,
    home_team_abbreviation=None,
    away_team_abbreviation=None,
):
    return {
        "game_id": game_id,
        "season": season,
        "date": pd.Timestamp(date, tz="America/New_York"),
        "home_team_id": home_team_id,
        "home_team_abbreviation": home_team_abbreviation or f"H{home_team_id}",
        "home_score": home_score,
        "away_team_id": away_team_id,
        "away_team_abbreviation": away_team_abbreviation or f"A{away_team_id}",
        "away_score": away_score,
        "home_winner": home_score > away_score,
        "away_winner": away_score > home_score,
    }


def _schedules_df(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _betting_line_row(**overrides):
    row = {
        "season": 2019,
        "game_date": pd.Timestamp("2020-01-01"),
        "home_team_abbrev": "CLE",
        "visit_team_abbrev": "BOS",
        "line": -4.5,
        "spread": 4.5,
    }
    row.update(overrides)
    return row


def _betting_lines_df(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def test_always_home_is_always_true():
    schedules = _schedules_df(
        [
            _schedule_row(1, "2020-01-01", 2020, TEAM_A, TEAM_B, 100, 90),
            _schedule_row(2, "2020-01-02", 2020, TEAM_B, TEAM_A, 80, 110),
        ]
    )

    result = baselines.always_home(schedules)

    assert result.name == "predicted_home_win"
    assert result.dtype == "boolean"
    assert list(result.index) == [1, 2]
    assert result.tolist() == [True, True]


def test_better_record_picks_the_better_record_even_when_away():
    schedules = _schedules_df(
        [
            # Team A builds a 2-0 record against Team C.
            _schedule_row(1, "2020-01-01", 2020, TEAM_A, TEAM_C, 110, 90),
            _schedule_row(2, "2020-01-03", 2020, TEAM_A, TEAM_C, 105, 95),
            # Team B builds a 0-2 record against Team D.
            _schedule_row(3, "2020-01-01", 2020, TEAM_D, TEAM_B, 100, 90),
            _schedule_row(4, "2020-01-03", 2020, TEAM_D, TEAM_B, 100, 90),
            # Team B hosts Team A: A (2-0, away) should be picked over B (0-2, home).
            _schedule_row(5, "2020-01-06", 2020, TEAM_B, TEAM_A, 90, 100),
        ]
    )

    result = baselines.better_record(schedules)

    assert result.dtype == "boolean"
    assert bool(result.loc[5]) is False


def test_better_record_opening_day_tie_defaults_to_home():
    schedules = _schedules_df(
        [
            _schedule_row(1, "2020-01-01", 2020, TEAM_A, TEAM_B, 100, 95),
        ]
    )

    result = baselines.better_record(schedules)

    assert bool(result.loc[1]) is True


def test_better_record_is_scoped_to_season():
    schedules = _schedules_df(
        [
            # Team A won every game last season...
            _schedule_row(1, "2019-01-01", 2019, TEAM_A, TEAM_C, 110, 90),
            _schedule_row(2, "2019-01-03", 2019, TEAM_A, TEAM_C, 110, 90),
            # ...but that record shouldn't carry into the new season, so
            # this opening-day matchup is a 0.5-vs-0.5 tie, defaulting home.
            _schedule_row(3, "2020-01-01", 2020, TEAM_B, TEAM_A, 100, 95),
        ]
    )

    result = baselines.better_record(schedules)

    assert bool(result.loc[3]) is True


def test_elo_alone_home_advantage_flips_a_small_raw_rating_gap():
    schedules = _schedules_df(
        [
            # Team B (id 2) upsets Team C (id 3) on the road, nudging B's
            # raw rating a bit above the still-untouched initial rating.
            _schedule_row(1, "2020-01-01", 2020, TEAM_C, TEAM_B, 90, 100),
            # Team A (id 1, still at the initial rating) hosts Team B, whose
            # raw rating is now higher than A's -- but by less than the
            # default 100-point home_advantage.
            _schedule_row(2, "2020-01-03", 2020, TEAM_A, TEAM_B, 0, 0),
        ]
    )

    with_full_home_advantage = baselines.elo_alone(schedules, home_advantage=100.0)
    assert bool(with_full_home_advantage.loc[2]) is True

    with_no_home_advantage = baselines.elo_alone(schedules, home_advantage=0.0)
    assert bool(with_no_home_advantage.loc[2]) is False


def test_elo_alone_output_dtype_and_index():
    schedules = _schedules_df(
        [
            _schedule_row(1, "2020-01-01", 2020, TEAM_A, TEAM_B, 110, 90),
        ]
    )

    result = baselines.elo_alone(schedules)

    assert result.dtype == "boolean"
    assert list(result.index) == [1]


def test_vegas_favorite_negative_line_picks_home():
    schedules = _schedules_df(
        [
            _schedule_row(
                1, "2020-01-01", 2020, TEAM_A, TEAM_B, 0, 0,
                home_team_abbreviation="CLE", away_team_abbreviation="BOS",
            ),
        ]
    )
    betting_lines = _betting_lines_df(
        [
            _betting_line_row(
                game_date="2020-01-01", home_team_abbrev="CLE", visit_team_abbrev="BOS", line=-4.5
            )
        ]
    )

    result = baselines.vegas_favorite(schedules, betting_lines)

    assert bool(result.loc[1]) is True


def test_vegas_favorite_positive_line_picks_away():
    schedules = _schedules_df(
        [
            _schedule_row(
                1, "2020-01-02", 2020, TEAM_A, TEAM_B, 0, 0,
                home_team_abbreviation="BOS", away_team_abbreviation="CLE",
            ),
        ]
    )
    betting_lines = _betting_lines_df(
        [
            _betting_line_row(
                game_date="2020-01-02", home_team_abbrev="BOS", visit_team_abbrev="CLE", line=3.5
            )
        ]
    )

    result = baselines.vegas_favorite(schedules, betting_lines)

    assert bool(result.loc[1]) is False


def test_vegas_favorite_pick_em_line_is_na():
    schedules = _schedules_df(
        [
            _schedule_row(
                1, "2020-01-03", 2020, TEAM_A, TEAM_B, 0, 0,
                home_team_abbreviation="CLE", away_team_abbreviation="BOS",
            ),
        ]
    )
    betting_lines = _betting_lines_df(
        [
            _betting_line_row(
                game_date="2020-01-03", home_team_abbrev="CLE", visit_team_abbrev="BOS", line=0
            )
        ]
    )

    result = baselines.vegas_favorite(schedules, betting_lines)

    assert result.loc[1] is pd.NA


def test_vegas_favorite_normalizes_mismatched_abbreviations():
    schedules = _schedules_df(
        [
            _schedule_row(
                1, "2020-01-04", 2020, TEAM_A, TEAM_B, 0, 0,
                home_team_abbreviation="GS", away_team_abbreviation="BOS",
            ),
        ]
    )
    betting_lines = _betting_lines_df(
        [
            _betting_line_row(
                game_date="2020-01-04", home_team_abbrev="GSW", visit_team_abbrev="BOS", line=-2.0
            )
        ]
    )

    result = baselines.vegas_favorite(schedules, betting_lines)

    assert bool(result.loc[1]) is True


def test_vegas_favorite_unmatched_game_is_na():
    schedules = _schedules_df(
        [
            _schedule_row(
                1, "2020-01-05", 2020, TEAM_A, TEAM_B, 0, 0,
                home_team_abbreviation="CLE", away_team_abbreviation="BOS",
            ),
        ]
    )
    betting_lines = _betting_lines_df(
        [
            _betting_line_row(
                game_date="2020-01-01", home_team_abbrev="CLE", visit_team_abbrev="BOS", line=-4.5
            )
        ]
    )

    result = baselines.vegas_favorite(schedules, betting_lines)

    assert result.loc[1] is pd.NA
