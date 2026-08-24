"""Production training entrypoint (PRD §14/§19).

New *usage* of Phase 1's `train()` (`models/train.py`, unchanged), not new
training logic - `through_date` is always "now" and seasons always the full
`config.season_range()`, unlike the backtest's per-fold cutoffs. The result
is persisted via T5.3's repository and immediately made active, so a
retrain either becomes the serving model or is a saved version an admin can
roll back to.

Activation also refreshes every pending (not-yet-played) prediction to the
newly active model - a live-tracking forecast for a game that hasn't
happened yet has no accuracy record to protect, so there's no reason to
leave it stale after a retrain (see `prediction_tracking`).
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.orm import Session

from nba_game_projections import config
from nba_game_projections.app.repositories.model_versions import (
    ModelVersionRecord,
    build_model_version_repository,
)
from nba_game_projections.app.services.prediction_tracking import capture_pending_predictions
from nba_game_projections.models.train import train


def run_production_training(session: Session) -> ModelVersionRecord:
    """Train on all data through now, persist the result, activate it, and refresh forecasts."""
    trained_model = train(pd.Timestamp.now(), seasons=config.season_range())

    repository = build_model_version_repository(session)
    record = repository.save(trained_model)
    activated = repository.activate(record.id)
    capture_pending_predictions(session)
    return activated
