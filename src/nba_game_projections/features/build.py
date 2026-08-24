"""Assembles per-team-per-game training-ready feature rows (T2.5).

Combines T2.1 (roster-aware trailing rolling stats, with optional T2.4
manual roster override), T2.2 (schedule-derived context), and T2.3 (Elo
ratings) into one feature matrix, one row per team per game, for games
whose date falls in `[start_date, end_date)`.

Context and Elo are computed once over the *full* loaded schedule history
(not just the target range) so rest-day state and Elo ratings correctly
carry forward from games before the range — only the output rows are
filtered down to the target range afterward. Rolling stats have no such
bulk form; they're inherently per-(team, as-of-date) and computed with one
call per target row (acceptable for a Phase 1 POC; not optimized for the
row counts a full walk-forward backtest will eventually need).

The `schedules` loader's `date` is timezone-aware while `player_box`'s
`game_date` is timezone-naive (see their respective modules); this module
strips the timezone off `schedules` once at the top so every date
comparison downstream — including against a caller's `start_date`/
`end_date` and a roster-override file's `prediction_date` — is naive and
directly comparable.

Deliberately does not attach a training target (margin of victory) — that
join against `schedules`' scores is simple and left to whoever builds T3.1,
keeping this module scoped to exactly the three feature sources named
above.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from nba_game_projections.data.player_box import load_player_box
from nba_game_projections.data.schedules import load_schedules
from nba_game_projections.features.context import build_team_game_context
from nba_game_projections.features.elo import compute_elo_ratings
from nba_game_projections.features.rolling_stats import team_rolling_features, team_roster
from nba_game_projections.features.roster_overrides import (
    RosterOverride,
    apply_roster_overrides,
    load_roster_overrides,
)

_ROLLING_FEATURE_KEYS = (
    "points",
    "rebounds",
    "assists",
    "steals",
    "blocks",
    "turnovers",
    "efg_pct",
    "minutes_per_game",
    "roster_size",
)


def _rolling_features_row(
    player_box: pd.DataFrame,
    team_id: int,
    as_of_date: pd.Timestamp,
    window: int,
    overrides: list[RosterOverride],
) -> dict[str, float]:
    automatic_roster = team_roster(player_box, team_id, as_of_date, window)
    active_ids = apply_roster_overrides(team_id, automatic_roster, overrides)
    features = team_rolling_features(
        player_box, team_id, as_of_date, window=window, active_player_ids=active_ids
    )
    return {f"rolling_{key}": features[key] for key in _ROLLING_FEATURE_KEYS}


def build_feature_matrix(
    start_date,
    end_date,
    seasons: Iterable[int] | None = None,
    rolling_window: int = 10,
    elo_k_factor: float = 20.0,
    elo_initial_rating: float = 1500.0,
    elo_home_advantage: float = 100.0,
    overrides: list[RosterOverride] | None = None,
    schedules_override: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the per-team-per-game feature matrix for games in `[start_date, end_date)`.

    `seasons` defaults to `config.season_range()` via the Wave 1 loaders
    (used for `player_box` regardless of `schedules_override`, below).

    Manual roster overrides: by default (`overrides=None`), loads
    `config.ROSTER_OVERRIDES_PATH` (T2.4) and applies it only to rows whose
    game date matches the file's single `prediction_date` — the file
    describes one prediction date's worth of edits, not a standing
    adjustment across a historical range. If `overrides` is given instead
    (Phase 2's live, "current state" SQLite overrides — T7.2), it applies
    to every row in the range unconditionally, with no date matching -
    there's no `prediction_date` in that model.

    `schedules_override`, if given, is used in place of calling
    `load_schedules(seasons)` - already-naive `date`, with at least
    `game_id`, `season`, `date`, `home_team_id`, `away_team_id`,
    `home_score`, `away_score`. This exists for Phase 2's prediction serving
    (T8.1), which needs feature rows for games that haven't been played yet
    - `load_schedules()` can't read a not-yet-started season's cached file
    at all (T5.4/PRD §16), so it's never asked to; the caller instead
    supplies real completed history plus the target games as placeholder
    rows (their pre-game Elo/context features don't depend on their own
    not-yet-real score).
    """
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    if schedules_override is not None:
        schedules = schedules_override
    else:
        schedules = load_schedules(seasons)
        schedules = schedules.assign(date=schedules["date"].dt.tz_localize(None))
    player_box = load_player_box(seasons)

    context = build_team_game_context(schedules)
    in_range = (context["date"] >= start_date) & (context["date"] < end_date)
    context = context.loc[in_range].reset_index(drop=True)

    elo = compute_elo_ratings(
        schedules,
        k_factor=elo_k_factor,
        initial_rating=elo_initial_rating,
        home_advantage=elo_home_advantage,
    ).rename(columns={"pre_game_rating": "elo_rating"})
    elo_in_range = (elo["date"] >= start_date) & (elo["date"] < end_date)
    elo = elo.loc[elo_in_range, ["game_id", "team_id", "elo_rating"]]

    if overrides is not None:
        live_overrides = overrides
        prediction_date = None
    else:
        prediction_date, live_overrides = load_roster_overrides()

    rolling_rows = []
    for row in context.itertuples(index=False):
        as_of_date = row.date
        if overrides is not None:
            row_overrides = live_overrides
        else:
            row_overrides = live_overrides if as_of_date.date() == prediction_date else []
        rolling_rows.append(
            {
                "game_id": row.game_id,
                "team_id": row.team_id,
                **_rolling_features_row(
                    player_box, row.team_id, as_of_date, rolling_window, row_overrides
                ),
            }
        )
    rolling = pd.DataFrame(rolling_rows)

    matrix = context.merge(elo, on=["game_id", "team_id"], how="left")
    matrix = matrix.merge(rolling, on=["game_id", "team_id"], how="left")
    return matrix
