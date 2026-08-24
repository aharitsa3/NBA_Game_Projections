import pandas as pd
import pytest

from nba_game_projections.data import team_box


def _fixture_frame(season: int, n: int = 2) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "game_id": 400000000 + i,
                "season": season,
                "season_type": 2,
                "game_date": f"{season - 1}-11-0{i + 1}",
                "game_date_time": pd.Timestamp(f"{season - 1}-11-0{i + 1}T19:00:00"),
                "team_id": 1 + i,
                "team_uid": f"s:40~l:46~t:{1 + i}",
                "team_slug": "team-slug",
                "team_location": "City",
                "team_name": "Team",
                "team_abbreviation": "TM",
                "team_display_name": "City Team",
                "team_short_display_name": "Team",
                "team_color": "000000",
                "team_alternate_color": "ffffff",
                "team_logo": "http://example.com/logo.png",
                "team_home_away": "home" if i % 2 == 0 else "away",
                "team_score": 100 + i,
                "team_winner": i % 2 == 0,
                "assists": 20 + i,
                "blocks": 5,
                "defensive_rebounds": 30,
                "fast_break_points": "12",
                "field_goal_pct": 0.45,
                "field_goals_made": 38,
                "field_goals_attempted": 85,
                "flagrant_fouls": 0,
                "fouls": 18,
                "free_throw_pct": 0.75,
                "free_throws_made": 15,
                "free_throws_attempted": 20,
                "largest_lead": "10",
                "offensive_rebounds": 10,
                "points_in_paint": "40",
                "steals": 7,
                "team_turnovers": 12,
                "technical_fouls": 0,
                "three_point_field_goal_pct": 0.35,
                "three_point_field_goals_made": 10,
                "three_point_field_goals_attempted": 28,
                "total_rebounds": 40,
                "total_technical_fouls": 0,
                "total_turnovers": 12,
                "turnover_points": "14",
                "turnovers": 12,
                "opponent_team_id": 2 - i,
                "opponent_team_uid": f"s:40~l:46~t:{2 - i}",
                "opponent_team_slug": "opp-slug",
                "opponent_team_location": "OppCity",
                "opponent_team_name": "OppTeam",
                "opponent_team_abbreviation": "OPP",
                "opponent_team_display_name": "OppCity OppTeam",
                "opponent_team_short_display_name": "OppTeam",
                "opponent_team_color": "111111",
                "opponent_team_alternate_color": "eeeeee",
                "opponent_team_logo": "http://example.com/opp-logo.png",
                "opponent_team_score": 95 + i,
            }
        )
    return pd.DataFrame(rows)


def _write_season(tmp_path, season: int, n: int = 2):
    dest = tmp_path / "team_box" / f"team_box_{season}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    _fixture_frame(season, n).to_parquet(dest)
    return dest


def test_load_team_box_schema_and_dtypes(monkeypatch, tmp_path):
    monkeypatch.setattr(team_box.config, "DATA_RAW_DIR", tmp_path)
    _write_season(tmp_path, 2020, n=3)

    df = team_box.load_team_box([2020])

    assert len(df) == 3
    assert list(df.columns) == team_box._COLUMNS
    for column, dtype in team_box._DTYPES.items():
        assert str(df[column].dtype) == dtype, column
    assert pd.api.types.is_datetime64_any_dtype(df["game_date"])


def test_load_team_box_concatenates_multiple_seasons(monkeypatch, tmp_path):
    monkeypatch.setattr(team_box.config, "DATA_RAW_DIR", tmp_path)
    _write_season(tmp_path, 2019, n=2)
    _write_season(tmp_path, 2020, n=3)

    df = team_box.load_team_box([2019, 2020])

    assert len(df) == 5
    assert set(df["season"]) == {2019, 2020}


def test_load_team_box_defaults_to_season_range(monkeypatch, tmp_path):
    monkeypatch.setattr(team_box.config, "DATA_RAW_DIR", tmp_path)
    monkeypatch.setattr(team_box.config, "season_range", lambda: range(2020, 2021))
    _write_season(tmp_path, 2020, n=1)

    df = team_box.load_team_box()

    assert len(df) == 1
    assert set(df["season"]) == {2020}


def test_missing_season_raises_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setattr(team_box.config, "DATA_RAW_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="2021"):
        team_box.load_team_box([2021])
