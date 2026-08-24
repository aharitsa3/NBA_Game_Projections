"""Loads the cached hoopR `betting_lines` JSON archive into a validated DataFrame.

Reads the single archive cached by `data/download.py` at
`data/raw/betting_lines/games-archive.json` (~7500 flat game records, not
season-partitioned within the file, unlike the other Wave 1 loaders) and
selects/casts it down to the closing-spread / over-under / cover-result
schema used later only for reporting baselines (PRD §8, T4.2/T4.3) - this
data is never a training feature or target (PRD §2, §8).

There is no explicit `game_id` tying a row here to the hoopR
`schedules`/`player_box`/`team_box` `game_id`, so `(season, game_date,
home_team_abbrev, visit_team_abbrev)` is kept as the natural join key for
whoever builds T4.2, rather than inventing a join in this module.

Both `line` (signed, relative to the home team - negative means the home
team is favored) and `spread` (the unsigned magnitude of that same line)
are kept, since they carry different information and it's not this
module's call which one downstream reporting code will want.

`season`, `home_team_id`, and `visit_team_id` come through the raw JSON as
strings even though they're numeric - cast to int here. `game_over_under`
similarly arrives as a numeric string ("216.0") - cast to float via
`pd.to_numeric(..., errors="coerce")` rather than `astype`, since a small
number of real archive rows carry `""` for a line that was never posted
(verified against the live archive) and should become `NaN`, not raise.
`name` is a raw HTML snippet used for display on the source site and is
dropped, as are the other purely cosmetic/redundant fields (`game_time`,
`tipoff`, `month`, `start`, `score`, `total`).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from nba_game_projections import config

_COLUMNS = [
    "season",
    "game_date",
    "home_team_id",
    "home_team_stats_id",
    "home_team_abbrev",
    "visit_team_id",
    "visit_team_stats_id",
    "visit_team_abbrev",
    "home_team_score",
    "visit_team_score",
    "game_over_under",
    "line",
    "spread",
    "favorite",
    "favorite_covered",
    "underdog_covered",
    "over_hit",
    "under_hit",
]

_INT_COLUMNS = [
    "season",
    "home_team_id",
    "visit_team_id",
    "home_team_score",
    "visit_team_score",
]

_FLOAT_COLUMNS = ["game_over_under", "line", "spread"]

_BOOL_COLUMNS = ["favorite_covered", "underdog_covered", "over_hit", "under_hit"]


def _archive_path() -> Path:
    return config.DATA_RAW_DIR / "betting_lines" / "games-archive.json"


def load_betting_lines(seasons: Iterable[int] | None = None) -> pd.DataFrame:
    """Load the cached betting_lines archive into one validated DataFrame.

    Unlike the other Wave 1 loaders, the source is a single unpartitioned
    JSON archive covering every season from 2017, so there's no per-season
    file to select - `seasons`, if given, filters the loaded rows down to
    that subset after loading rather than choosing which file(s) to read.

    Raises `FileNotFoundError` if the archive hasn't been downloaded yet -
    this function only reads the local cache, it never downloads.
    """
    path = _archive_path()
    if not path.exists():
        raise FileNotFoundError(
            f"No cached betting_lines archive at {path}. "
            "Run `download.download_betting_lines()` (or `download.download_all()`) first."
        )

    records = json.loads(path.read_text())
    df = pd.DataFrame(records, columns=_COLUMNS)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df[_INT_COLUMNS] = df[_INT_COLUMNS].astype("int64")
    # A handful of real archive rows carry "" instead of a number for lines
    # that were never posted (e.g. game_over_under on a small number of
    # games) - coerce rather than astype so those become NaN instead of
    # raising.
    df[_FLOAT_COLUMNS] = df[_FLOAT_COLUMNS].apply(pd.to_numeric, errors="coerce")
    df[_BOOL_COLUMNS] = df[_BOOL_COLUMNS].astype(bool)

    if seasons is not None:
        df = df[df["season"].isin(list(seasons))].reset_index(drop=True)

    return df
