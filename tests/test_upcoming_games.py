import pandas as pd
import pytest

from nba_game_projections.data import upcoming_games


def _future_season_df(season: int) -> pd.DataFrame:
    """Shaped like the real 2026-27 schedule file (PRD §16): no home_winner/
    away_winner columns at all, placeholder 0 scores, status_type_completed
    is the only "played yet?" signal - a mix of played and unplayed rows.
    """
    return pd.DataFrame(
        {
            "game_id": [500_000, 500_001, 500_002],
            "season": [season] * 3,
            "season_type": [2] * 3,
            "game_date_time": pd.to_datetime(
                ["2026-10-21 19:30:00", "2026-10-22 19:30:00", "2026-10-23 19:30:00"], utc=True
            ),
            "home_id": [1610612737, 1610612738, 1610612739],
            "home_display_name": ["Hawks", "Celtics", "Nets"],
            "away_id": [1610612740, 1610612741, 1610612742],
            "away_display_name": ["Pelicans", "Bulls", "Cavaliers"],
            "home_score": [0, 0, 105],
            "away_score": [0, 0, 98],
            "status_type_completed": [False, False, True],
        }
    )


def _completed_season_df(season: int) -> pd.DataFrame:
    """Shaped like a fully-completed season's file: real scores/winners,
    status_type_completed True throughout.
    """
    return pd.DataFrame(
        {
            "game_id": [400_000, 400_001],
            "season": [season] * 2,
            "season_type": [2] * 2,
            "game_date_time": pd.to_datetime(
                ["2025-01-05 19:30:00", "2025-01-06 19:30:00"], utc=True
            ),
            "home_id": [1610612737, 1610612738],
            "home_display_name": ["Hawks", "Celtics"],
            "away_id": [1610612740, 1610612741],
            "away_display_name": ["Pelicans", "Bulls"],
            "home_score": [110, 99],
            "away_score": [102, 101],
            "home_winner": [True, False],
            "away_winner": [False, True],
            "status_type_completed": [True, True],
        }
    )


def test_load_upcoming_games_returns_only_unplayed_rows(monkeypatch, tmp_path):
    monkeypatch.setattr(upcoming_games.config, "DATA_RAW_DIR", tmp_path)
    dest = tmp_path / "schedules" / "nba_schedule_2027.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    _future_season_df(2027).to_parquet(dest)

    df = upcoming_games.load_upcoming_games(season=2027)

    assert list(df.columns) == [
        "game_id",
        "season",
        "season_type",
        "date",
        "home_team_id",
        "home_team_name",
        "away_team_id",
        "away_team_name",
    ]
    assert df["game_id"].tolist() == [500_000, 500_001]
    assert df["home_team_name"].tolist() == ["Hawks", "Celtics"]


def test_load_upcoming_games_dtypes(monkeypatch, tmp_path):
    monkeypatch.setattr(upcoming_games.config, "DATA_RAW_DIR", tmp_path)
    dest = tmp_path / "schedules" / "nba_schedule_2027.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    _future_season_df(2027).to_parquet(dest)

    df = upcoming_games.load_upcoming_games(season=2027)

    assert df["game_id"].dtype == "int32"
    assert df["home_team_id"].dtype == "int32"
    assert df["away_team_id"].dtype == "int32"


def test_load_upcoming_games_on_completed_season_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(upcoming_games.config, "DATA_RAW_DIR", tmp_path)
    dest = tmp_path / "schedules" / "nba_schedule_2025.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    _completed_season_df(2025).to_parquet(dest)

    df = upcoming_games.load_upcoming_games(season=2025)

    assert df.empty


def test_load_upcoming_games_defaults_to_current_season(monkeypatch, tmp_path):
    monkeypatch.setattr(upcoming_games.config, "DATA_RAW_DIR", tmp_path)
    monkeypatch.setattr(upcoming_games.config, "current_season", lambda: 2027)
    dest = tmp_path / "schedules" / "nba_schedule_2027.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    _future_season_df(2027).to_parquet(dest)

    df = upcoming_games.load_upcoming_games()

    assert (df["season"] == 2027).all()


def test_missing_season_file_raises_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setattr(upcoming_games.config, "DATA_RAW_DIR", tmp_path)

    with pytest.raises(FileNotFoundError) as excinfo:
        upcoming_games.load_upcoming_games(season=2099)

    message = str(excinfo.value)
    assert "2099" in message
    assert "nba_schedule_2099.parquet" in message
