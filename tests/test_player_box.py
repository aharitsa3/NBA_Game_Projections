import pandas as pd
import pytest

from nba_game_projections.data import player_box

_RAW_COLUMNS = [
    "game_id",
    "season",
    "season_type",
    "game_date",
    "game_date_time",
    "athlete_id",
    "athlete_display_name",
    "team_id",
    "team_name",
    "minutes",
    "field_goals_made",
    "field_goals_attempted",
    "three_point_field_goals_made",
    "three_point_field_goals_attempted",
    "free_throws_made",
    "free_throws_attempted",
    "offensive_rebounds",
    "defensive_rebounds",
    "rebounds",
    "assists",
    "steals",
    "blocks",
    "turnovers",
    "fouls",
    "plus_minus",
    "points",
    "starter",
    "ejected",
    "did_not_play",
    "reason",
    "active",
    "home_away",
    "team_winner",
    "team_score",
    "opponent_team_id",
    "opponent_team_score",
]


def _raw_row(**overrides):
    row = {
        "game_id": 401234567,
        "season": 2020,
        "season_type": 2,
        "game_date": "2020-01-15",
        "game_date_time": pd.Timestamp("2020-01-15 19:30:00"),
        "athlete_id": 1001,
        "athlete_display_name": "Test Player",
        "team_id": 5,
        "team_name": "Testers",
        "minutes": 32.0,
        "field_goals_made": 8.0,
        "field_goals_attempted": 15.0,
        "three_point_field_goals_made": 2.0,
        "three_point_field_goals_attempted": 5.0,
        "free_throws_made": 4.0,
        "free_throws_attempted": 5.0,
        "offensive_rebounds": 1.0,
        "defensive_rebounds": 4.0,
        "rebounds": 5.0,
        "assists": 6.0,
        "steals": 1.0,
        "blocks": 0.0,
        "turnovers": 2.0,
        "fouls": 3.0,
        "plus_minus": "+7",
        "points": 22.0,
        "starter": True,
        "ejected": False,
        "did_not_play": False,
        "reason": None,
        "active": True,
        "home_away": "home",
        "team_winner": True,
        "team_score": 110,
        "opponent_team_id": 9,
        "opponent_team_score": 103,
    }
    row.update(overrides)
    return row


def _write_season(tmp_path, season, rows):
    dest_dir = tmp_path / "player_box"
    dest_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=_RAW_COLUMNS)
    df.to_parquet(dest_dir / f"player_box_{season}.parquet")


def test_load_player_box_schema_and_dtypes(monkeypatch, tmp_path):
    monkeypatch.setattr(player_box.config, "DATA_RAW_DIR", tmp_path)

    rows = [
        _raw_row(),
        _raw_row(
            athlete_id=1002,
            athlete_display_name="Benched Player",
            starter=False,
            did_not_play=True,
            reason="COACH'S DECISION",
            minutes=0.0,
            points=0.0,
        ),
    ]
    _write_season(tmp_path, 2020, rows)

    result = player_box.load_player_box(seasons=[2020])

    assert len(result) == 2
    expected_columns = set(player_box._COLUMNS.values())
    assert set(result.columns) == expected_columns

    assert result["game_id"].dtype.kind == "i"
    assert result["athlete_id"].dtype.kind == "i"
    assert result["team_id"].dtype.kind == "i"
    assert result["minutes"].dtype.kind == "f"
    assert result["points"].dtype.kind == "f"
    assert result["starter"].dtype == bool
    assert result["did_not_play"].dtype == bool
    assert pd.api.types.is_datetime64_any_dtype(result["game_date"])

    starter_row = result.loc[result["athlete_id"] == 1001].iloc[0]
    assert starter_row["starter"]
    assert not starter_row["did_not_play"]
    assert pd.isna(starter_row["dnp_reason"])

    dnp_row = result.loc[result["athlete_id"] == 1002].iloc[0]
    assert not dnp_row["starter"]
    assert dnp_row["did_not_play"]
    assert dnp_row["dnp_reason"] == "COACH'S DECISION"


def test_load_player_box_concatenates_multiple_seasons(monkeypatch, tmp_path):
    monkeypatch.setattr(player_box.config, "DATA_RAW_DIR", tmp_path)

    _write_season(tmp_path, 2018, [_raw_row(season=2018, game_id=1)])
    _write_season(tmp_path, 2019, [_raw_row(season=2019, game_id=2)])

    result = player_box.load_player_box(seasons=[2018, 2019])

    assert len(result) == 2
    assert set(result["season"]) == {2018, 2019}


def test_load_player_box_defaults_to_season_range(monkeypatch, tmp_path):
    monkeypatch.setattr(player_box.config, "DATA_RAW_DIR", tmp_path)
    monkeypatch.setattr(player_box.config, "season_range", lambda: range(2020, 2022))

    _write_season(tmp_path, 2020, [_raw_row(season=2020)])
    _write_season(tmp_path, 2021, [_raw_row(season=2021, game_id=2)])

    result = player_box.load_player_box()

    assert set(result["season"]) == {2020, 2021}


def test_missing_season_raises_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setattr(player_box.config, "DATA_RAW_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="2020"):
        player_box.load_player_box(seasons=[2020])
