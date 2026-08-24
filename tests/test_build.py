import json

import pandas as pd

from nba_game_projections.features import build, roster_overrides

_SCHEDULE_COLUMNS = [
    "game_id",
    "season",
    "season_type",
    "date",
    "home_team_id",
    "home_team_name",
    "home_team_abbreviation",
    "home_score",
    "away_team_id",
    "away_team_name",
    "away_team_abbreviation",
    "away_score",
    "home_winner",
    "away_winner",
]

_PLAYER_BOX_COLUMNS = [
    "game_id",
    "season",
    "game_date",
    "athlete_id",
    "athlete_display_name",
    "team_id",
    "home_away",
    "opponent_team_id",
    "starter",
    "did_not_play",
    "dnp_reason",
    "minutes",
    "points",
    "rebounds",
    "offensive_rebounds",
    "defensive_rebounds",
    "assists",
    "steals",
    "blocks",
    "turnovers",
    "fouls",
    "field_goals_made",
    "field_goals_attempted",
    "three_point_field_goals_made",
    "three_point_field_goals_attempted",
    "free_throws_made",
    "free_throws_attempted",
]


def _ts(date_str: str) -> pd.Timestamp:
    return pd.Timestamp(date_str, tz="America/New_York")


def _schedule_row(**overrides):
    row = {
        "game_id": 1,
        "season": 2020,
        "season_type": 2,
        "date": _ts("2020-11-01"),
        "home_team_id": 1,
        "home_team_name": "Team One",
        "home_team_abbreviation": "T1",
        "home_score": 100,
        "away_team_id": 2,
        "away_team_name": "Team Two",
        "away_team_abbreviation": "T2",
        "away_score": 90,
        "home_winner": True,
        "away_winner": False,
    }
    row.update(overrides)
    return row


def _schedules_fixture() -> pd.DataFrame:
    rows = [
        _schedule_row(
            game_id=1, season=2020, date=_ts("2020-11-01"), home_team_id=1, away_team_id=2
        ),
        _schedule_row(
            game_id=2, season=2020, date=_ts("2020-11-05"), home_team_id=2, away_team_id=1
        ),
        _schedule_row(
            game_id=3, season=2020, date=_ts("2020-12-01"), home_team_id=1, away_team_id=2
        ),
        # target game, inside [2021-01-01, 2021-01-06)
        _schedule_row(
            game_id=4, season=2021, date=_ts("2021-01-05"), home_team_id=2, away_team_id=1
        ),
        # future game, outside the target range - must never leak into game 4's features
        _schedule_row(
            game_id=5, season=2021, date=_ts("2021-01-10"), home_team_id=1, away_team_id=2,
            home_score=999, away_score=1,
        ),
    ]
    return pd.DataFrame(rows, columns=_SCHEDULE_COLUMNS)


def _player_row(**overrides):
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


def _player_games(athlete_id, team_id, game_ids, dates, **stat_overrides):
    return [
        _player_row(
            game_id=gid,
            athlete_id=athlete_id,
            team_id=team_id,
            game_date=pd.Timestamp(date),
            **stat_overrides,
        )
        for gid, date in zip(game_ids, dates, strict=True)
    ]


def _player_box_fixture() -> pd.DataFrame:
    common_game_ids = [1, 2, 3, 4]
    common_dates = ["2020-11-01", "2020-11-05", "2020-12-01", "2021-01-05"]

    rows = []
    rows += _player_games(
        11, 1, common_game_ids, common_dates,
        points=20.0, rebounds=5.0, assists=3.0, minutes=30.0,
        field_goals_made=8.0, field_goals_attempted=15.0, three_point_field_goals_made=2.0,
    )
    rows += _player_games(
        12, 1, common_game_ids, common_dates,
        points=10.0, rebounds=6.0, assists=2.0, minutes=25.0,
        field_goals_made=4.0, field_goals_attempted=10.0, three_point_field_goals_made=1.0,
    )
    rows += _player_games(
        21, 2, common_game_ids, common_dates,
        points=15.0, rebounds=4.0, assists=5.0, minutes=28.0,
        field_goals_made=6.0, field_goals_attempted=12.0, three_point_field_goals_made=2.0,
    )
    rows += _player_games(
        22, 2, common_game_ids, common_dates,
        points=12.0, rebounds=3.0, assists=4.0, minutes=20.0,
        field_goals_made=5.0, field_goals_attempted=11.0, three_point_field_goals_made=1.0,
    )
    # future poison game (2021-01-10): wild stats that must never reach game 4's features
    rows += _player_games(11, 1, [5], ["2021-01-10"], points=999.0, minutes=999.0)
    rows += _player_games(21, 2, [5], ["2021-01-10"], points=999.0, minutes=999.0)

    return pd.DataFrame(rows, columns=_PLAYER_BOX_COLUMNS)


def _patch_loaders(monkeypatch):
    monkeypatch.setattr(build, "load_schedules", lambda seasons=None: _schedules_fixture())
    monkeypatch.setattr(build, "load_player_box", lambda seasons=None: _player_box_fixture())


def test_feature_matrix_shape_and_columns(monkeypatch, tmp_path):
    _patch_loaders(monkeypatch)
    monkeypatch.setattr(roster_overrides.config, "ROSTER_OVERRIDES_PATH", tmp_path / "missing.json")

    matrix = build.build_feature_matrix("2021-01-01", "2021-01-06", rolling_window=3)

    assert len(matrix) == 2  # one row per side of game 4 only
    assert set(matrix["game_id"]) == {4}

    expected_columns = {
        "game_id", "date", "season", "team_id", "opponent_team_id", "is_home",
        "rest_days", "is_back_to_back", "elo_rating",
        "rolling_points", "rolling_rebounds", "rolling_assists", "rolling_steals",
        "rolling_blocks", "rolling_turnovers", "rolling_efg_pct",
        "rolling_minutes_per_game", "rolling_roster_size",
    }
    assert expected_columns.issubset(set(matrix.columns))


def test_no_leakage_from_future_game(monkeypatch, tmp_path):
    _patch_loaders(monkeypatch)
    monkeypatch.setattr(roster_overrides.config, "ROSTER_OVERRIDES_PATH", tmp_path / "missing.json")

    matrix = build.build_feature_matrix("2021-01-01", "2021-01-06", rolling_window=3)

    team1_row = matrix.loc[matrix["team_id"] == 1].iloc[0]
    team2_row = matrix.loc[matrix["team_id"] == 2].iloc[0]

    # team 1's trailing 3 games before 2021-01-05 are games 1/2/3 only - the
    # 999-point future game (2021-01-10) must never be included.
    assert team1_row["rolling_points"] == 30.0  # player 11 (20) + player 12 (10)
    assert team2_row["rolling_points"] == 27.0  # player 21 (15) + player 22 (12)
    assert team1_row["rolling_minutes_per_game"] < 900


def test_context_and_elo_are_populated_and_causal(monkeypatch, tmp_path):
    _patch_loaders(monkeypatch)
    monkeypatch.setattr(roster_overrides.config, "ROSTER_OVERRIDES_PATH", tmp_path / "missing.json")

    matrix = build.build_feature_matrix("2021-01-01", "2021-01-06", rolling_window=3)

    # both teams last played game 3 on 2020-12-01, 35 days before game 4
    assert (matrix["rest_days"] == 35).all()
    assert not matrix["is_back_to_back"].any()

    # both teams have prior game history, so ratings should have moved off
    # the 1500.0 initial default by the time of game 4
    assert not (matrix["elo_rating"] == 1500.0).any()


def test_roster_override_excludes_player_on_matching_prediction_date(monkeypatch, tmp_path):
    _patch_loaders(monkeypatch)
    overrides_path = tmp_path / "roster_overrides.json"
    overrides_path.write_text(
        json.dumps(
            {
                "prediction_date": "2021-01-05",
                "overrides": [
                    {
                        "player_id": 11,
                        "player_name": "Player Eleven",
                        "team_id": 1,
                        "status": "out",
                        "note": "test override",
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(roster_overrides.config, "ROSTER_OVERRIDES_PATH", overrides_path)

    matrix = build.build_feature_matrix("2021-01-01", "2021-01-06", rolling_window=3)

    team1_row = matrix.loc[matrix["team_id"] == 1].iloc[0]
    assert team1_row["rolling_points"] == 10.0  # player 12 only, player 11 excluded
    assert team1_row["rolling_roster_size"] == 1


def test_override_does_not_apply_outside_its_prediction_date(monkeypatch, tmp_path):
    _patch_loaders(monkeypatch)
    overrides_path = tmp_path / "roster_overrides.json"
    overrides_path.write_text(
        json.dumps(
            {
                "prediction_date": "2020-11-01",  # does not match game 4's date
                "overrides": [
                    {
                        "player_id": 11,
                        "player_name": "Player Eleven",
                        "team_id": 1,
                        "status": "out",
                        "note": "test override",
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(roster_overrides.config, "ROSTER_OVERRIDES_PATH", overrides_path)

    matrix = build.build_feature_matrix("2021-01-01", "2021-01-06", rolling_window=3)

    team1_row = matrix.loc[matrix["team_id"] == 1].iloc[0]
    assert team1_row["rolling_points"] == 30.0  # override not in effect for this date


def test_explicit_overrides_apply_unconditionally_to_every_row(monkeypatch, tmp_path):
    """Phase 2's live overrides (T7.2) have no `prediction_date` to match against -
    passing `overrides` explicitly applies it to every row in range, unlike the
    file-based default's date-matching behavior."""
    _patch_loaders(monkeypatch)
    monkeypatch.setattr(roster_overrides.config, "ROSTER_OVERRIDES_PATH", tmp_path / "missing.json")

    live_overrides = [
        roster_overrides.RosterOverride(
            player_id=11, player_name="Player Eleven", team_id=1, status="out"
        )
    ]

    matrix = build.build_feature_matrix(
        "2021-01-01", "2021-01-06", rolling_window=3, overrides=live_overrides
    )

    team1_row = matrix.loc[matrix["team_id"] == 1].iloc[0]
    assert team1_row["rolling_points"] == 10.0  # player 12 only, player 11 excluded
    assert team1_row["rolling_roster_size"] == 1


def test_empty_explicit_overrides_list_applies_no_overrides(monkeypatch, tmp_path):
    _patch_loaders(monkeypatch)
    overrides_path = tmp_path / "roster_overrides.json"
    overrides_path.write_text(
        json.dumps(
            {
                "prediction_date": "2021-01-05",
                "overrides": [
                    {
                        "player_id": 11,
                        "player_name": "Player Eleven",
                        "team_id": 1,
                        "status": "out",
                        "note": "should be ignored - explicit overrides takes precedence",
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(roster_overrides.config, "ROSTER_OVERRIDES_PATH", overrides_path)

    matrix = build.build_feature_matrix(
        "2021-01-01", "2021-01-06", rolling_window=3, overrides=[]
    )

    team1_row = matrix.loc[matrix["team_id"] == 1].iloc[0]
    assert team1_row["rolling_points"] == 30.0  # file-based override never consulted


def test_schedules_override_bypasses_load_schedules(monkeypatch, tmp_path):
    """T8.1's use case: a not-yet-started season's file can't be read by
    `load_schedules()` at all (PRD §16), so `schedules_override` must let a
    caller skip that call entirely rather than merely stub its output."""

    def _fail_if_called(seasons=None):
        raise AssertionError("load_schedules() should not be called when schedules_override is set")

    monkeypatch.setattr(build, "load_schedules", _fail_if_called)
    monkeypatch.setattr(build, "load_player_box", lambda seasons=None: _player_box_fixture())
    monkeypatch.setattr(roster_overrides.config, "ROSTER_OVERRIDES_PATH", tmp_path / "missing.json")

    naive_schedule = _schedules_fixture().assign(date=lambda df: df["date"].dt.tz_localize(None))

    matrix = build.build_feature_matrix(
        "2021-01-01", "2021-01-06", rolling_window=3, schedules_override=naive_schedule
    )

    assert len(matrix) == 2
    assert set(matrix["game_id"]) == {4}
