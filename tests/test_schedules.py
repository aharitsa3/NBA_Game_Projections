import datetime

import pandas as pd
import pytest

from nba_game_projections.data import schedules


def _raw_schedule_df(season: int, n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "status_type_name": ["STATUS_FINAL"] * n,
            "game_id": [400_000_000 + season * 100 + i for i in range(n)],
            "season": [season] * n,
            "season_type": [2] * n,
            "date": [f"{season}-01-0{i + 1}" for i in range(n)],
            "game_date": [f"{season}-01-0{i + 1}" for i in range(n)],
            "game_date_time": pd.to_datetime(
                [f"{season}-01-0{i + 1} 19:30:00" for i in range(n)], utc=True
            ).tz_convert("America/New_York"),
            "home_id": [1610612737 + i for i in range(n)],
            "home_display_name": [f"Home Team {i}" for i in range(n)],
            "home_abbreviation": [f"H{i}" for i in range(n)],
            "home_score": [100 + i for i in range(n)],
            "away_id": [1610612738 + i for i in range(n)],
            "away_display_name": [f"Away Team {i}" for i in range(n)],
            "away_abbreviation": [f"A{i}" for i in range(n)],
            "away_score": [95 + i for i in range(n)],
            "home_winner": [True, False, True][:n],
            "away_winner": [False, True, False][:n],
            "venue_full_name": [f"Arena {i}" for i in range(n)],
            "broadcast_market": ["home"] * n,
        }
    )


def _write_season(tmp_path, season: int, n: int = 3) -> None:
    dest = tmp_path / "schedules" / f"nba_schedule_{season}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    _raw_schedule_df(season, n).to_parquet(dest)


def test_load_schedules_schema_and_dtypes(monkeypatch, tmp_path):
    monkeypatch.setattr(schedules.config, "DATA_RAW_DIR", tmp_path)
    _write_season(tmp_path, 2020, n=3)

    df = schedules.load_schedules(seasons=[2020])

    assert list(df.columns) == [
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
    assert len(df) == 3

    assert df["game_id"].dtype == "int32"
    assert df["season"].dtype == "int32"
    assert df["season_type"].dtype == "int32"
    assert df["home_team_id"].dtype == "int32"
    assert df["away_team_id"].dtype == "int32"
    assert df["home_score"].dtype == "int32"
    assert df["away_score"].dtype == "int32"
    assert df["home_winner"].dtype == bool
    assert df["away_winner"].dtype == bool
    assert isinstance(df["date"].dtype, pd.DatetimeTZDtype)

    assert (df["season"] == 2020).all()
    assert df["home_team_name"].tolist() == ["Home Team 0", "Home Team 1", "Home Team 2"]


def test_load_schedules_concatenates_multiple_seasons(monkeypatch, tmp_path):
    monkeypatch.setattr(schedules.config, "DATA_RAW_DIR", tmp_path)
    _write_season(tmp_path, 2019, n=2)
    _write_season(tmp_path, 2020, n=3)

    df = schedules.load_schedules(seasons=[2019, 2020])

    assert len(df) == 5
    assert sorted(df["season"].unique().tolist()) == [2019, 2020]


def test_load_schedules_defaults_to_season_range(monkeypatch, tmp_path):
    monkeypatch.setattr(schedules.config, "DATA_RAW_DIR", tmp_path)

    class FixedDate(datetime.date):
        @classmethod
        def today(cls):
            return cls(2020, 3, 1)

    monkeypatch.setattr(schedules.config.datetime, "date", FixedDate)
    monkeypatch.setattr(schedules.config, "SEASON_START", 2019)

    _write_season(tmp_path, 2019, n=1)
    _write_season(tmp_path, 2020, n=1)

    df = schedules.load_schedules()

    assert sorted(df["season"].unique().tolist()) == [2019, 2020]


def test_missing_season_file_raises_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setattr(schedules.config, "DATA_RAW_DIR", tmp_path)

    with pytest.raises(FileNotFoundError) as excinfo:
        schedules.load_schedules(seasons=[2021])

    message = str(excinfo.value)
    assert "2021" in message
    assert "nba_schedule_2021.parquet" in message


def test_load_schedules_in_progress_season_without_winner_columns(monkeypatch, tmp_path):
    monkeypatch.setattr(schedules.config, "DATA_RAW_DIR", tmp_path)
    raw = _raw_schedule_df(2027, n=3).drop(columns=["home_winner", "away_winner"])
    raw["status_type_name"] = ["STATUS_FINAL", "STATUS_FINAL", "STATUS_SCHEDULED"]
    raw["home_score"] = [110, 90, 0]
    raw["away_score"] = [100, 95, 0]
    dest = tmp_path / "schedules" / "nba_schedule_2027.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(dest)

    df = schedules.load_schedules(seasons=[2027])

    assert len(df) == 2
    assert df["home_winner"].tolist() == [True, False]
    assert df["away_winner"].tolist() == [False, True]


def test_load_schedules_drops_scheduled_games_even_with_winner_columns(monkeypatch, tmp_path):
    monkeypatch.setattr(schedules.config, "DATA_RAW_DIR", tmp_path)
    raw = _raw_schedule_df(2027, n=3)
    raw["status_type_name"] = ["STATUS_FINAL", "STATUS_SCHEDULED", "STATUS_SCHEDULED"]
    dest = tmp_path / "schedules" / "nba_schedule_2027.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(dest)

    assert len(schedules.load_schedules(seasons=[2027])) == 1
