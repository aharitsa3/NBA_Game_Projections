"""Loads cached hoopR `team_box` parquet files into a validated DataFrame.

One row per team per game: game identity, team score/result, and the core
aggregate box-score stats (rebounds, assists, steals, blocks, turnovers,
fouls, shooting splits). Cosmetic columns (colors, logos, slugs, uids) and
free-text opponent metadata are dropped — only `opponent_team_id` and
`opponent_team_score` are kept, for convenience in downstream joins.

This module only reads what `download.py` has already cached; it never
fetches from the network.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from nba_game_projections import config

_COLUMNS = [
    "game_id",
    "season",
    "season_type",
    "game_date",
    "team_id",
    "team_home_away",
    "team_score",
    "team_winner",
    "assists",
    "blocks",
    "steals",
    "turnovers",
    "fouls",
    "offensive_rebounds",
    "defensive_rebounds",
    "total_rebounds",
    "field_goals_made",
    "field_goals_attempted",
    "field_goal_pct",
    "three_point_field_goals_made",
    "three_point_field_goals_attempted",
    "three_point_field_goal_pct",
    "free_throws_made",
    "free_throws_attempted",
    "free_throw_pct",
    "opponent_team_id",
    "opponent_team_score",
]

_DTYPES = {
    "game_id": "int32",
    "season": "int32",
    "season_type": "int32",
    "team_id": "int32",
    "team_home_away": "object",
    "team_score": "int32",
    "team_winner": "bool",
    "assists": "int32",
    "blocks": "int32",
    "steals": "int32",
    "turnovers": "int32",
    "fouls": "int32",
    "offensive_rebounds": "int32",
    "defensive_rebounds": "int32",
    "total_rebounds": "int32",
    "field_goals_made": "int32",
    "field_goals_attempted": "int32",
    "field_goal_pct": "float64",
    "three_point_field_goals_made": "int32",
    "three_point_field_goals_attempted": "int32",
    "three_point_field_goal_pct": "float64",
    "free_throws_made": "int32",
    "free_throws_attempted": "int32",
    "free_throw_pct": "float64",
    "opponent_team_id": "int32",
    "opponent_team_score": "int32",
}


def _season_path(season: int) -> Path:
    return config.DATA_RAW_DIR / "team_box" / f"team_box_{season}.parquet"


def _load_season(season: int) -> pd.DataFrame:
    path = _season_path(season)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached team_box file for season {season} at {path}. "
            "Run `download.download_season_files('team_box', [...])` first."
        )
    df = pd.read_parquet(path, columns=_COLUMNS)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df.astype(_DTYPES)


def load_team_box(seasons: Iterable[int] | None = None) -> pd.DataFrame:
    """Load and concatenate cached team_box files for `seasons` (default: `season_range()`)."""
    seasons = seasons if seasons is not None else config.season_range()
    frames = [_load_season(season) for season in seasons]
    return pd.concat(frames, ignore_index=True)
