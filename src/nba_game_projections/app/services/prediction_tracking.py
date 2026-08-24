"""Live prediction tracking: capture, reconcile, and score real predictions.

Two independent, explicitly-triggered actions (never bundled into
`refresh-data`, and kept separate from each other):

- `capture_pending_predictions` upserts a row for every not-yet-played
  game in the next 14 days via `prediction.predict_pending_games` (that
  horizon - not "every known upcoming game" - keeps the feature-matrix
  build tractable and the forecast meaningful; see that function's
  docstring) - inserting newly-visible games and refreshing predicted
  values for anything not yet reconciled (see
  `PredictionRepository.upsert_prediction`). Also run automatically right
  after a retrain activates a new model, so pending forecasts reflect it
  without a separate manual step.
- `reconcile_predictions` fills in actual results for pending rows whose
  game now appears completed in the *locally cached* schedule data - it
  never fetches; call `refresh-data` first to pick up newly finished games.

`compute_accuracy` then scores only reconciled rows - a prediction that
hasn't been measured against a real outcome yet says nothing about model
accuracy.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.orm import Session

from nba_game_projections import config
from nba_game_projections.app.repositories.model_versions import build_model_version_repository
from nba_game_projections.app.repositories.predictions import (
    PredictionRecord,
    build_prediction_repository,
)
from nba_game_projections.app.services.prediction import predict_pending_games
from nba_game_projections.backtest.report import brier_score
from nba_game_projections.data.schedules import load_schedules


def capture_pending_predictions(session: Session, horizon_days: int = 14) -> list[PredictionRecord]:
    """Insert/refresh a prediction row for every not-yet-played game in the next `horizon_days`.

    Raises `NoActiveModelError` (from `predict_pending_games`) if no model
    version has been activated yet.
    """
    predictions = predict_pending_games(session, horizon_days=horizon_days)
    if not predictions:
        return []

    active_model_id = build_model_version_repository(session).get_active().id
    repository = build_prediction_repository(session)
    return [
        repository.upsert_prediction(model_version_id=active_model_id, **prediction)
        for prediction in predictions
    ]


def reconcile_predictions(session: Session) -> list[PredictionRecord]:
    """Fill in actual results for pending predictions whose game is now complete locally."""
    repository = build_prediction_repository(session)
    pending = repository.list_pending()
    if not pending:
        return []

    completed = load_schedules(config.season_range()).set_index("game_id")

    reconciled = []
    for record in pending:
        if record.game_id not in completed.index:
            continue
        game = completed.loc[record.game_id]
        actual_margin = float(game["home_score"] - game["away_score"])
        actual_winner = "home" if bool(game["home_winner"]) else "away"
        updated = repository.reconcile(
            record.game_id, actual_margin=actual_margin, actual_winner=actual_winner
        )
        if updated is not None:
            reconciled.append(updated)
    return reconciled


def _summarize(group: pd.DataFrame) -> dict:
    error = group["predicted_margin"] - group["actual_margin"]
    return {
        "n_games": len(group),
        "accuracy": float((group["predicted_winner"] == group["actual_winner"]).mean()),
        "mae": float(error.abs().mean()),
        "rmse": float((error.pow(2).mean()) ** 0.5),
        "brier_score": brier_score(group["home_win_probability"], group["actual_home_win"]),
    }


def compute_accuracy(session: Session) -> dict:
    """Win/loss accuracy, margin MAE/RMSE, and Brier score over reconciled predictions.

    Reported both pooled and broken out per `model_version_id`, so a
    retrain's effect on accuracy is visible without a separate query.
    """
    reconciled = [
        record for record in build_prediction_repository(session).list() if record.reconciled_at
    ]
    if not reconciled:
        return {"pooled": None, "by_model_version": []}

    df = pd.DataFrame(
        {
            "model_version_id": [r.model_version_id for r in reconciled],
            "predicted_winner": [r.predicted_winner for r in reconciled],
            "actual_winner": [r.actual_winner for r in reconciled],
            "predicted_margin": [r.predicted_margin for r in reconciled],
            "actual_margin": [r.actual_margin for r in reconciled],
            "home_win_probability": [r.home_win_probability for r in reconciled],
        }
    )
    df["actual_home_win"] = df["actual_winner"] == "home"

    return {
        "pooled": _summarize(df),
        "by_model_version": [
            {"model_version_id": version_id, **_summarize(group)}
            for version_id, group in df.groupby("model_version_id")
        ],
    }
