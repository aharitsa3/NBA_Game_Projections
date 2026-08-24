"""Loads cached hoopR `schedules` parquet files into a validated DataFrame.

Each cached file (`data/raw/schedules/nba_schedule_{season}.parquet`, one per
season, written by `download.py`) has 77 raw columns (venue, broadcast,
colors/logos, etc.). This module selects and renames just the columns this
pipeline needs — game id, date, home/away team, final score, season — down
to a clean, stable schema, rather than passing the raw columns through.

`game_date_time` is used as the `date` column (rather than the raw `date` or
`game_date` columns) since it's tz-aware and the most precise of the three.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from nba_game_projections import config

_RAW_TO_CLEAN = {
    "game_id": "game_id",
    "season": "season",
    "season_type": "season_type",
    "game_date_time": "date",
    "home_id": "home_team_id",
    "home_display_name": "home_team_name",
    "home_abbreviation": "home_team_abbreviation",
    "home_score": "home_score",
    "away_id": "away_team_id",
    "away_display_name": "away_team_name",
    "away_abbreviation": "away_team_abbreviation",
    "away_score": "away_score",
    "home_winner": "home_winner",
    "away_winner": "away_winner",
}

_CLEAN_COLUMNS = list(_RAW_TO_CLEAN.values())

_INT32_COLUMNS = [
    "game_id",
    "season",
    "season_type",
    "home_team_id",
    "home_score",
    "away_team_id",
    "away_score",
]


def _season_path(season: int) -> Path:
    return config.DATA_RAW_DIR / "schedules" / f"nba_schedule_{season}.parquet"


def _load_season(season: int) -> pd.DataFrame:
    path = _season_path(season)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached schedules file for season {season} at {path}. "
            "Run `download.download_season_files('schedules', [...])` "
            "(or `download.download_all()`) first."
        )

    raw = pd.read_parquet(path, columns=list(_RAW_TO_CLEAN))
    df = raw.rename(columns=_RAW_TO_CLEAN)[_CLEAN_COLUMNS]
    df[_INT32_COLUMNS] = df[_INT32_COLUMNS].astype("int32")
    df["home_winner"] = df["home_winner"].astype(bool)
    df["away_winner"] = df["away_winner"].astype(bool)
    return df


def load_schedules(seasons: Iterable[int] | None = None) -> pd.DataFrame:
    """Load and concatenate cached schedules files into one validated DataFrame.

    Defaults to `config.season_range()` when `seasons` is None. Raises
    `FileNotFoundError` for any requested season that hasn't been downloaded
    yet — this function only reads the local cache, it never downloads.
    """
    seasons = list(seasons) if seasons is not None else list(config.season_range())
    frames = [_load_season(season) for season in seasons]
    return pd.concat(frames, ignore_index=True)
