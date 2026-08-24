"""Expanding-window walk-forward backtest harness (T4.1, PRD §8).

Trains on all seasons strictly before a cutoff, tests on the next season,
rolls the cutoff forward one season at a time. No random/k-fold splits -
those would leak future information into training via rolling-window
features (PRD §8).

**First valid test season (PRD §11 open item, resolved here):** a season
needs at least one full prior season to train on at all (an empty training
set isn't a meaningful fold), so the first test season is the second
season in `seasons` by default (`min_train_seasons=1`). That single-prior-
season requirement is the actual binding constraint, not the rolling
window size - a 10-game rolling window needs far less trailing history
than a whole season provides.

Each fold's test-season features are built via `build_feature_matrix` with
`seasons` including the test season itself, alongside every training
season - this is necessary (not leakage): Elo ratings and rest days for a
mid-season test game must reflect that season's *own* earlier games, whose
results are already known by prediction time. `train()`'s own `seasons`
argument, by contrast, is deliberately restricted to strictly-prior
seasons so the model itself never learns from the test season's outcomes.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from nba_game_projections import config
from nba_game_projections.data.schedules import load_schedules
from nba_game_projections.features.build import build_feature_matrix
from nba_game_projections.models.train import FEATURE_COLUMNS, train, wide_game_matrix
from nba_game_projections.models.win_probability import win_probability

_OUTPUT_COLUMNS = [
    "test_season",
    "game_id",
    "date",
    "home_team_id",
    "away_team_id",
    "home_score",
    "away_score",
    "actual_margin",
    "actual_home_win",
    "predicted_margin",
    "predicted_home_win",
    "predicted_win_probability",
]


def _season_window(schedules: pd.DataFrame, season: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    dates = schedules.loc[schedules["season"] == season, "date"]
    return dates.min(), dates.max() + pd.Timedelta(days=1)


def run_walk_forward(
    seasons: Iterable[int] | None = None,
    min_train_seasons: int = 1,
    rolling_window: int = 10,
    elo_k_factor: float = 20.0,
    elo_initial_rating: float = 1500.0,
    elo_home_advantage: float = 100.0,
    model_params: dict | None = None,
) -> pd.DataFrame:
    """Run the expanding-window backtest and return one row per test-fold game.

    `seasons` defaults to `config.season_range()`. Returns predicted and
    actual margins/win outcomes plus a derived win probability (T3.2, using
    each fold's own training residuals) for every game across every test
    fold - the model and residuals are refit from scratch each fold, never
    reused across folds.
    """
    seasons = list(seasons) if seasons is not None else list(config.season_range())
    if len(seasons) <= min_train_seasons:
        raise ValueError(
            f"Need more than {min_train_seasons} season(s) to have any test fold; got {seasons}"
        )

    schedules = load_schedules(seasons)
    schedules = schedules.assign(date=schedules["date"].dt.tz_localize(None))

    fold_frames = []
    for i in range(min_train_seasons, len(seasons)):
        test_season = seasons[i]
        training_seasons = seasons[:i]
        test_start, test_end = _season_window(schedules, test_season)

        model = train(
            through_date=test_start,
            seasons=training_seasons,
            rolling_window=rolling_window,
            elo_k_factor=elo_k_factor,
            elo_initial_rating=elo_initial_rating,
            elo_home_advantage=elo_home_advantage,
            model_params=model_params,
        )

        long_test = build_feature_matrix(
            test_start,
            test_end,
            seasons=[*training_seasons, test_season],
            rolling_window=rolling_window,
            elo_k_factor=elo_k_factor,
            elo_initial_rating=elo_initial_rating,
            elo_home_advantage=elo_home_advantage,
        )
        wide_test = wide_game_matrix(long_test)
        assert (wide_test["date"] >= test_start).all() and (wide_test["date"] < test_end).all(), (
            f"leakage: fold for test_season={test_season} pulled in a row outside its own season"
        )

        wide_test = wide_test.merge(
            schedules[["game_id", "home_score", "away_score"]], on="game_id"
        )
        wide_test["actual_margin"] = wide_test["home_score"] - wide_test["away_score"]
        wide_test["actual_home_win"] = wide_test["actual_margin"] > 0

        predicted_margin = model.predict(wide_test[list(FEATURE_COLUMNS)])
        wide_test["predicted_margin"] = predicted_margin
        wide_test["predicted_home_win"] = predicted_margin > 0
        wide_test["predicted_win_probability"] = [
            win_probability(margin, model.residuals) for margin in predicted_margin
        ]
        wide_test["test_season"] = test_season

        fold_frames.append(wide_test[_OUTPUT_COLUMNS])

    return pd.concat(fold_frames, ignore_index=True)
