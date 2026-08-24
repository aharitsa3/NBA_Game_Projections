"""Loads a season's not-yet-played games from the cached `schedules` file.

Per PRD §16, a not-yet-completed season's cached file
(`nba_schedule_{season}.parquet`) has a different shape than a completed
one: no `home_winner`/`away_winner` columns at all, and placeholder `0`
scores for every row. `status_type_completed=False` is the reliable
"not yet played" signal instead. This is deliberately a new, separate
function from `data/schedules.py`'s `load_schedules()` - that loader is not
touched, so its "every row is a completed game with a real score"
guarantee (depended on by every Wave 1-4 caller) stays intact.
"""

from __future__ import annotations

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
    "away_id": "away_team_id",
    "away_display_name": "away_team_name",
    "status_type_completed": "status_type_completed",
}

_CLEAN_COLUMNS = [c for c in _RAW_TO_CLEAN.values() if c != "status_type_completed"]

_INT32_COLUMNS = ["game_id", "season", "season_type", "home_team_id", "away_team_id"]


def _season_path(season: int) -> Path:
    return config.DATA_RAW_DIR / "schedules" / f"nba_schedule_{season}.parquet"


def load_upcoming_games(season: int | None = None) -> pd.DataFrame:
    """Return `season`'s not-yet-played games (game_id, date, home/away team).

    Defaults to `config.current_season()`. Reads only the local cache,
    same as `load_schedules()` - raises `FileNotFoundError` if the season's
    file hasn't been downloaded yet.
    """
    season = season if season is not None else config.current_season()
    path = _season_path(season)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached schedules file for season {season} at {path}. "
            "Run `download.download_season_files('schedules', [...])` "
            "(or `download.download_all()`) first."
        )

    raw = pd.read_parquet(path, columns=list(_RAW_TO_CLEAN))
    unplayed = raw[~raw["status_type_completed"].astype(bool)]
    df = unplayed.rename(columns=_RAW_TO_CLEAN)[_CLEAN_COLUMNS].reset_index(drop=True)
    df[_INT32_COLUMNS] = df[_INT32_COLUMNS].astype("int32")
    return df
