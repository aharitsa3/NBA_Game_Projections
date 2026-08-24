"""Builds team-level, schedule-derived contextual features per team per game.

Reshapes the one-row-per-game output of `data.schedules.load_schedules` into
one row per team per game, then computes each team's rest days and
back-to-back flag from its own strictly-earlier games — naturally causal,
since a team's rest state never depends on any other team's schedule.
"""

from __future__ import annotations

import pandas as pd

_HOME_COLUMNS = {
    "home_team_id": "team_id",
    "away_team_id": "opponent_team_id",
}
_AWAY_COLUMNS = {
    "away_team_id": "team_id",
    "home_team_id": "opponent_team_id",
}

_OUTPUT_COLUMNS = [
    "game_id",
    "date",
    "season",
    "team_id",
    "opponent_team_id",
    "is_home",
    "rest_days",
    "is_back_to_back",
]


def build_team_game_context(schedules: pd.DataFrame) -> pd.DataFrame:
    """Reshape `schedules` into one row per team per game with rest/B2B features.

    Returns columns: `game_id`, `date`, `season`, `team_id`,
    `opponent_team_id`, `is_home`, `rest_days`, `is_back_to_back`.

    `rest_days` is the number of calendar days since that team's previous
    game (NaN for a team's first game in the dataset); `is_back_to_back` is
    True when `rest_days == 1`.
    """
    base_columns = ["game_id", "date", "season"]
    home = schedules[[*base_columns, "home_team_id", "away_team_id"]].rename(
        columns=_HOME_COLUMNS
    )
    home["is_home"] = True
    away = schedules[[*base_columns, "home_team_id", "away_team_id"]].rename(
        columns=_AWAY_COLUMNS
    )
    away["is_home"] = False

    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["team_id", "date"], kind="stable").reset_index(drop=True)

    # `date` carries a tipoff time-of-day, so a game the evening before would
    # otherwise diff to a fractional day; normalize to calendar dates first.
    game_date = long["date"].dt.normalize()
    prev_game_date = game_date.groupby(long["team_id"]).shift(1)
    long["rest_days"] = (game_date - prev_game_date).dt.days
    long["is_back_to_back"] = long["rest_days"] == 1

    return long[_OUTPUT_COLUMNS]
