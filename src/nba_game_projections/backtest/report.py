"""Metrics and final comparison report (T4.3, PRD §8).

Consumes `walk_forward.run_walk_forward`'s output schema directly (one row
per test-fold game: `test_season`, `game_id`, `home_team_id`, `away_team_id`,
`actual_margin`, `actual_home_win`, `predicted_margin`, `predicted_home_win`,
`predicted_win_probability`) rather than calling the backtest itself, so
this module stays independently testable against small synthetic frames.

**Headline metric (PRD §8):** win/loss accuracy, per test season and pooled
across seasons, for the model and all four baselines (T4.2) side by side.
Baseline predictions use the nullable `"boolean"` dtype and can be `pd.NA`
(a Vegas pick'em, or a game the Vegas join couldn't match) - those rows are
excluded from that baseline's own accuracy denominator rather than guessed.

**Secondary metrics (PRD §3, §8):** margin MAE/RMSE (model only - baselines
don't predict a margin) and win-probability calibration (Brier score plus a
binned reliability table), both model-only for the same reason.
"""

from __future__ import annotations

import pandas as pd

from nba_game_projections.backtest import baselines

_PREDICTOR_COLUMNS = {
    "model": "predicted_home_win",
    "always_home": "baseline_always_home",
    "better_record": "baseline_better_record",
    "elo_alone": "baseline_elo_alone",
    "vegas_favorite": "baseline_vegas_favorite",
}


def _win_rate_stats(predicted: pd.Series, actual: pd.Series) -> dict:
    mask = predicted.notna()
    predicted = predicted[mask].astype(bool)
    actual = actual[mask].astype(bool)
    n = len(predicted)
    correct = int((predicted == actual).sum())
    return {
        "n_games": n,
        "n_correct": correct,
        "accuracy": correct / n if n else float("nan"),
    }


def accuracy_table(
    df: pd.DataFrame,
    predictor_columns: dict[str, str],
    actual_col: str = "actual_home_win",
    season_col: str = "test_season",
) -> pd.DataFrame:
    """Win/loss accuracy per season and pooled, for each named predictor column."""
    rows = []
    for predictor, col in predictor_columns.items():
        for season, group in df.groupby(season_col):
            rows.append(
                {
                    "predictor": predictor,
                    "test_season": season,
                    **_win_rate_stats(group[col], group[actual_col]),
                }
            )
        rows.append(
            {
                "predictor": predictor,
                "test_season": "pooled",
                **_win_rate_stats(df[col], df[actual_col]),
            }
        )
    return pd.DataFrame(rows)


def _margin_error_stats(predicted_margin: pd.Series, actual_margin: pd.Series) -> dict:
    error = predicted_margin - actual_margin
    return {
        "n_games": len(error),
        "mae": error.abs().mean(),
        "rmse": (error.pow(2).mean()) ** 0.5,
    }


def margin_error_table(
    df: pd.DataFrame,
    predicted_col: str = "predicted_margin",
    actual_col: str = "actual_margin",
    season_col: str = "test_season",
) -> pd.DataFrame:
    """Model-only margin MAE/RMSE per season and pooled."""
    rows = [
        {"test_season": season, **_margin_error_stats(group[predicted_col], group[actual_col])}
        for season, group in df.groupby(season_col)
    ]
    rows.append(
        {"test_season": "pooled", **_margin_error_stats(df[predicted_col], df[actual_col])}
    )
    return pd.DataFrame(rows)


def brier_score(predicted_prob: pd.Series, actual: pd.Series) -> float:
    """Mean squared error between predicted win probability and the 0/1 outcome."""
    return float(((predicted_prob - actual.astype(int)) ** 2).mean())


def calibration_table(
    predicted_prob: pd.Series, actual: pd.Series, n_bins: int = 10
) -> pd.DataFrame:
    """Bins predicted win probability and compares to the realized win rate per bin."""
    bins = pd.cut(predicted_prob, bins=n_bins, include_lowest=True)
    frame = pd.DataFrame(
        {"predicted_prob": predicted_prob.to_numpy(), "actual": actual.astype(int).to_numpy()},
        index=bins,
    )
    table = frame.groupby(level=0, observed=True).agg(
        n_games=("actual", "size"),
        predicted_prob_mean=("predicted_prob", "mean"),
        actual_win_rate=("actual", "mean"),
    )
    return table.reset_index(names="bin")


def attach_baselines(
    predictions: pd.DataFrame, schedules: pd.DataFrame, betting_lines: pd.DataFrame
) -> pd.DataFrame:
    """Join each T4.2 baseline's pick onto `predictions` by `game_id`.

    Baseline predictions are computed over the *full* `schedules` history
    passed in (they're inherently causal, not fold-scoped - see
    `baselines.py`); assigning by a `game_id`-indexed Series naturally
    aligns down to just `predictions`' own rows.
    """
    out = predictions.set_index("game_id")
    out["baseline_always_home"] = baselines.always_home(schedules)
    out["baseline_better_record"] = baselines.better_record(schedules)
    out["baseline_elo_alone"] = baselines.elo_alone(schedules)
    out["baseline_vegas_favorite"] = baselines.vegas_favorite(schedules, betting_lines)
    return out.reset_index()


def build_report(
    predictions: pd.DataFrame, schedules: pd.DataFrame, betting_lines: pd.DataFrame
) -> dict[str, pd.DataFrame | float]:
    """Assemble the final model-vs-baselines comparison report."""
    with_baselines = attach_baselines(predictions, schedules, betting_lines)
    return {
        "accuracy": accuracy_table(with_baselines, _PREDICTOR_COLUMNS),
        "margin_error": margin_error_table(predictions),
        "brier_score": brier_score(
            predictions["predicted_win_probability"], predictions["actual_home_win"]
        ),
        "calibration": calibration_table(
            predictions["predicted_win_probability"], predictions["actual_home_win"]
        ),
    }
