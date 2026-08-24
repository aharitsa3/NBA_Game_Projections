"""Runs the full walk-forward backtest (T4.1) and comparison report (T4.3) over
`config.season_range()` (2017-present) and prints the results.

Requires `data/raw/{schedules,player_box,betting_lines}` to already be
downloaded (`nba_game_projections.data.download`) - this script only reads
the local cache, it never downloads.

Slow: the rolling-stats feature computation rescans the full player_box
table per team-game row (see `features/rolling_stats.py`/`features/build.py`
docstrings - a known, documented Phase 1 limitation), so later folds with
years of accumulated training data get progressively slower. Budget well
over an hour for the full 2017+ range.

Usage: `uv run python scripts/run_backtest.py`
"""

from __future__ import annotations

import time

from nba_game_projections import config
from nba_game_projections.backtest import report
from nba_game_projections.backtest.walk_forward import run_walk_forward
from nba_game_projections.data.betting_lines import load_betting_lines
from nba_game_projections.data.schedules import load_schedules

_PREDICTIONS_PATH = config.DATA_PROCESSED_DIR / "backtest_predictions.parquet"


def main() -> None:
    seasons = list(config.season_range())
    print("seasons:", seasons, flush=True)

    t0 = time.time()
    predictions = run_walk_forward(seasons=seasons)
    print(
        "walk-forward done in", round(time.time() - t0, 1), "s, rows:", len(predictions),
        flush=True,
    )

    _PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(_PREDICTIONS_PATH)
    print("predictions saved to", _PREDICTIONS_PATH, flush=True)

    schedules = load_schedules(seasons)
    # betting_lines labels a season by its start year (e.g. 2018 for the
    # 2018-19 season); hoopR/schedules label it by end year - shift to match
    # (see backtest/baselines.py's vegas_favorite docstring for the full story).
    betting_lines = load_betting_lines([season - 1 for season in seasons])

    result = report.build_report(predictions, schedules, betting_lines)

    print(flush=True)
    print("=== ACCURACY (win/loss, headline metric) ===", flush=True)
    print(result["accuracy"].to_string(), flush=True)
    print(flush=True)
    print("=== MARGIN ERROR (model only) ===", flush=True)
    print(result["margin_error"].to_string(), flush=True)
    print(flush=True)
    print("brier_score:", result["brier_score"], flush=True)
    print(flush=True)
    print("=== CALIBRATION (model only) ===", flush=True)
    print(result["calibration"].to_string(), flush=True)


if __name__ == "__main__":
    main()
