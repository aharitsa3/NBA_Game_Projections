"""Slate prediction service (T8.1) - Phase 2's integration point.

For a given date: loads that day's not-yet-played games (T5.4), builds
features via the existing `build_feature_matrix` (T2.5) using the live
roster-override table (T7.2) instead of the file-based default, pairs
into model-input rows via `wide_game_matrix` (T5.5), runs inference
through the currently-active model (T5.3), and derives win probability via
`win_probability` (T3.2, unchanged). Computed fresh on every call - no
caching (PRD §17) - so a roster-override edit changes the very next call's
probabilities.

`predict_pending_games` is the same computation over every not-yet-played
game in the next `horizon_days` (default 14), rather than one date - the
shared engine behind both lives in `_predict_games`. It exists for live
prediction tracking (`app.services.prediction_tracking`), which needs a
prediction for anything pending soon, not just today's slate. Capped to a
near-term horizon rather than every known upcoming game (which can span
into next season, many months out): a prediction that far ahead is
meaningless (Elo/rolling-stats context can't be projected that far in
advance) and would build a feature matrix far too large for
`features/rolling_stats.py`'s documented per-row-rescan cost to handle.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.orm import Session

from nba_game_projections import config
from nba_game_projections.app.repositories.model_versions import build_model_version_repository
from nba_game_projections.app.services.roster_overrides import load_live_roster_overrides
from nba_game_projections.data.schedules import load_schedules
from nba_game_projections.data.upcoming_games import load_upcoming_games
from nba_game_projections.features.build import build_feature_matrix
from nba_game_projections.models.train import wide_game_matrix
from nba_game_projections.models.win_probability import win_probability

_UPCOMING_GAMES_COLUMNS = [
    "game_id",
    "season",
    "season_type",
    "date",
    "home_team_id",
    "home_team_name",
    "away_team_id",
    "away_team_name",
]

_SYNTHETIC_SCHEDULE_COLUMNS = [
    "game_id",
    "season",
    "season_type",
    "date",
    "home_team_id",
    "away_team_id",
    "home_score",
    "away_score",
]


class NoActiveModelError(RuntimeError):
    """No model version is active - train and activate one first (T6.1/T6.3/T6.4)."""


def _all_upcoming_games() -> pd.DataFrame:
    """Union this season's and next season's cached not-yet-played games.

    `config.current_season()` labels the season by its *ending* year, so
    during the off-season it names the just-finished (fully played)
    season, not the not-yet-started one that already has a published
    schedule (PRD §12) - checking both season labels covers that gap
    without needing an explicit season argument from the caller.
    """
    frames = []
    for season in (config.current_season(), config.current_season() + 1):
        try:
            loaded = load_upcoming_games(season=season)
        except FileNotFoundError:
            continue
        # Skip empty frames: concatenating one in can coerce the tz-aware
        # `date` column of an otherwise non-empty frame to plain `object`
        # dtype, breaking the `.dt` accessor used below.
        if not loaded.empty:
            frames.append(loaded)

    if not frames:
        return pd.DataFrame(columns=_UPCOMING_GAMES_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _upcoming_games_for_date(target_date: pd.Timestamp) -> pd.DataFrame:
    games = _all_upcoming_games()
    if games.empty:
        return games
    naive_date = games["date"].dt.tz_localize(None)
    matches = naive_date.dt.date == target_date.date()
    return games.loc[matches].reset_index(drop=True)


def _schedules_with_target_games(target_games: pd.DataFrame) -> pd.DataFrame:
    """Real completed history plus the target slate as placeholder rows.

    `load_schedules()` can't read a not-yet-started season's cached file at
    all (T5.4/PRD §16), so it's only ever asked for `config.season_range()`
    (seasons known to be safe); the target games' own placeholder
    `home_score`/`away_score` never feeds a real computation - a game's
    pre-game Elo rating and rest days depend only on strictly-earlier games.
    """
    completed = load_schedules(config.season_range())
    completed = completed.assign(date=completed["date"].dt.tz_localize(None))
    completed = completed.loc[~completed["game_id"].isin(target_games["game_id"])]

    target_rows = pd.DataFrame(
        {
            "game_id": target_games["game_id"],
            "season": target_games["season"],
            "season_type": target_games["season_type"],
            "date": target_games["date"].dt.tz_localize(None),
            "home_team_id": target_games["home_team_id"],
            "away_team_id": target_games["away_team_id"],
            "home_score": 0,
            "away_score": 0,
        }
    )

    return pd.concat(
        [completed[_SYNTHETIC_SCHEDULE_COLUMNS], target_rows[_SYNTHETIC_SCHEDULE_COLUMNS]],
        ignore_index=True,
    )


def _team_names(target_games: pd.DataFrame) -> pd.Series:
    home = target_games[["home_team_id", "home_team_name"]].rename(
        columns={"home_team_id": "team_id", "home_team_name": "team_name"}
    )
    away = target_games[["away_team_id", "away_team_name"]].rename(
        columns={"away_team_id": "team_id", "away_team_name": "team_name"}
    )
    return pd.concat([home, away]).drop_duplicates("team_id").set_index("team_id")["team_name"]


def _predict_games(session: Session, target_games: pd.DataFrame) -> list[dict]:
    """Predicted margin/winner/win-probability for every game in `target_games`.

    Raises `NoActiveModelError` if no model version has been activated yet.
    `target_games` may span multiple dates - the feature-matrix range is
    widened to cover all of them in one build, then the result is filtered
    back down to exactly `target_games`' own game ids. That filter matters:
    the widened window can also contain real completed games whose date
    falls inside it (e.g. a game that lingered "not yet played" from
    earlier in the season) - those aren't predictions we were asked for
    and have no placeholder row to key a prediction dict off of.
    """
    if target_games.empty:
        return []

    active_model = build_model_version_repository(session).load_active_model()
    if active_model is None:
        raise NoActiveModelError("No active model version - train and activate one first.")

    game_dates = target_games.set_index("game_id")["date"].dt.tz_localize(None)
    range_start = game_dates.min()
    range_end = game_dates.max() + pd.Timedelta(days=1)

    long = build_feature_matrix(
        range_start,
        range_end,
        seasons=config.season_range(),
        overrides=load_live_roster_overrides(session),
        schedules_override=_schedules_with_target_games(target_games),
    )
    wide = wide_game_matrix(long)
    wide = wide.loc[wide["game_id"].isin(target_games["game_id"])].reset_index(drop=True)
    team_names = _team_names(target_games)

    predicted_margins = active_model.predict(wide)

    predictions = []
    for row, predicted_margin in zip(wide.itertuples(index=False), predicted_margins, strict=True):
        predicted_margin = float(predicted_margin)
        predictions.append(
            {
                "game_id": row.game_id,
                "date": game_dates.loc[row.game_id].date().isoformat(),
                "home_team_id": row.home_team_id,
                "home_team_name": team_names.get(row.home_team_id),
                "away_team_id": row.away_team_id,
                "away_team_name": team_names.get(row.away_team_id),
                "predicted_margin": predicted_margin,
                "predicted_winner": "home" if predicted_margin >= 0 else "away",
                "home_win_probability": win_probability(predicted_margin, active_model.residuals),
            }
        )
    return predictions


def predict_slate(session: Session, date: str | pd.Timestamp | None = None) -> list[dict]:
    """Predicted margin/winner/win-probability for every game on `date` (defaults to today)."""
    target_date = pd.Timestamp(date) if date is not None else pd.Timestamp.now().normalize()
    return _predict_games(session, _upcoming_games_for_date(target_date))


def _pending_games_within_horizon(horizon: pd.Timedelta) -> pd.DataFrame:
    games = _all_upcoming_games()
    if games.empty:
        return games
    today = pd.Timestamp.now().normalize()
    naive_date = games["date"].dt.tz_localize(None)
    in_horizon = (naive_date >= today) & (naive_date < today + horizon)
    return games.loc[in_horizon].reset_index(drop=True)


def predict_pending_games(session: Session, horizon_days: int = 14) -> list[dict]:
    """Predicted margin/winner/win-probability for every not-yet-played game in the next
    `horizon_days` (default 14) - see the module docstring for why this is capped rather
    than covering every known upcoming game.
    """
    horizon = pd.Timedelta(days=horizon_days)
    return _predict_games(session, _pending_games_within_horizon(horizon))
