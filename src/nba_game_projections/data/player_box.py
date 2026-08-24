"""Loads cached hoopR `player_box` parquet files into a validated DataFrame.

Reads the season files cached by `data/download.py` at
`data/raw/player_box/player_box_{season}.parquet`, selects/renames the
raw hoopR columns down to the per-player per-game schema this pipeline
needs (game identity, player identity, team context, box-score stats,
starter flag, DNP reason), and concatenates across seasons. Does not
download — `download.py` owns that.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from nba_game_projections import config

_COLUMNS = {
    "game_id": "game_id",
    "season": "season",
    "game_date": "game_date",
    "athlete_id": "athlete_id",
    "athlete_display_name": "athlete_display_name",
    "team_id": "team_id",
    "home_away": "home_away",
    "opponent_team_id": "opponent_team_id",
    "starter": "starter",
    "did_not_play": "did_not_play",
    "reason": "dnp_reason",
    "minutes": "minutes",
    "points": "points",
    "rebounds": "rebounds",
    "offensive_rebounds": "offensive_rebounds",
    "defensive_rebounds": "defensive_rebounds",
    "assists": "assists",
    "steals": "steals",
    "blocks": "blocks",
    "turnovers": "turnovers",
    "fouls": "fouls",
    "field_goals_made": "field_goals_made",
    "field_goals_attempted": "field_goals_attempted",
    "three_point_field_goals_made": "three_point_field_goals_made",
    "three_point_field_goals_attempted": "three_point_field_goals_attempted",
    "free_throws_made": "free_throws_made",
    "free_throws_attempted": "free_throws_attempted",
}


def _load_season(season: int) -> pd.DataFrame:
    path = config.DATA_RAW_DIR / "player_box" / f"player_box_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"no cached player_box file for season {season} at {path}; "
            "run `nba_game_projections.data.download.download_all()` first"
        )
    raw = pd.read_parquet(path, columns=list(_COLUMNS))
    df = raw.rename(columns=_COLUMNS)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def load_player_box(seasons: Iterable[int] | None = None) -> pd.DataFrame:
    """Load and concatenate cached player_box seasons into one DataFrame.

    Defaults to `config.season_range()`. Raises `FileNotFoundError` for any
    requested season that hasn't been downloaded yet.
    """
    seasons = list(seasons) if seasons is not None else list(config.season_range())
    frames = [_load_season(season) for season in seasons]
    return pd.concat(frames, ignore_index=True)
