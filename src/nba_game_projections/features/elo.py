"""Game-by-game Elo rating engine (PRD §5, §8 baseline #3).

Ratings are updated sequentially in chronological game order across the
full history passed in (all `season_type` values included by default —
playoffs count too, per PRD §5 "updated game-by-game across the full
history"). The feature exposed per game is each team's PRE-game rating:
using the post-game (already updated) rating would leak that game's own
result into its own feature row.
"""

from __future__ import annotations

import pandas as pd


def elo_win_probability(
    home_rating: float, away_rating: float, home_advantage: float = 100.0
) -> float:
    """Expected probability the home team wins, given current ratings."""
    return 1 / (1 + 10 ** (-((home_rating + home_advantage) - away_rating) / 400))


def compute_elo_ratings(
    schedules: pd.DataFrame,
    k_factor: float = 20.0,
    initial_rating: float = 1500.0,
    home_advantage: float = 100.0,
) -> pd.DataFrame:
    """Compute pre-game Elo ratings for every team in every game.

    Returns a long-format DataFrame, one row per team per game, with
    columns `game_id`, `date`, `team_id`, `opponent_team_id`, `is_home`,
    `pre_game_rating`, ordered chronologically then by team.
    """
    games = schedules.sort_values("date", kind="stable")

    ratings: dict[int, float] = {}
    rows: list[dict] = []

    for game in games.itertuples(index=False):
        home_id = game.home_team_id
        away_id = game.away_team_id
        home_rating = ratings.get(home_id, initial_rating)
        away_rating = ratings.get(away_id, initial_rating)

        rows.append(
            {
                "game_id": game.game_id,
                "date": game.date,
                "team_id": home_id,
                "opponent_team_id": away_id,
                "is_home": True,
                "pre_game_rating": home_rating,
            }
        )
        rows.append(
            {
                "game_id": game.game_id,
                "date": game.date,
                "team_id": away_id,
                "opponent_team_id": home_id,
                "is_home": False,
                "pre_game_rating": away_rating,
            }
        )

        expected_home = elo_win_probability(home_rating, away_rating, home_advantage)
        actual_home = 1.0 if game.home_score > game.away_score else 0.0

        ratings[home_id] = home_rating + k_factor * (actual_home - expected_home)
        ratings[away_id] = away_rating - k_factor * (actual_home - expected_home)

    return pd.DataFrame(
        rows,
        columns=["game_id", "date", "team_id", "opponent_team_id", "is_home", "pre_game_rating"],
    )
