"""Admin endpoints: data refresh, retraining, and model/roster-override management.

Built up incrementally across Wave 6/7 (T6.2-T6.4, T7.1).
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from nba_game_projections import config
from nba_game_projections.app.db import SessionLocal, get_session
from nba_game_projections.app.repositories.model_versions import (
    ModelVersionRecord,
    build_model_version_repository,
)
from nba_game_projections.app.repositories.predictions import (
    PredictionRecord,
    build_prediction_repository,
)
from nba_game_projections.app.repositories.roster_overrides import (
    RosterOverrideRecord,
    build_roster_override_repository,
)
from nba_game_projections.app.services.prediction import NoActiveModelError
from nba_game_projections.app.services.prediction_tracking import (
    capture_pending_predictions,
    compute_accuracy,
    reconcile_predictions,
)
from nba_game_projections.app.services.training import run_production_training
from nba_game_projections.data import download

router = APIRouter(prefix="/admin", tags=["admin"])

_SEASON_FOLDERS = ("player_box", "team_box", "schedules")

# In-memory job registry: retrain runs are admin-triggered, single-user, and
# not expected to survive an app restart, so no persistence needed here -
# unlike model versions (T5.3), which must survive one.
_retrain_jobs: dict[str, dict[str, str | None]] = {}


@router.post("/refresh-data")
def refresh_data() -> dict:
    """Refresh the current season's cached hoopR data (PRD §16).

    Synchronous - `download_season_files` already skips already-cached
    files, so this is a handful of small HTTP calls expected to take
    seconds, unlike a retrain.
    """
    season = config.current_season()
    downloaded = {
        folder: [str(path) for path in download.download_season_files(folder, [season])]
        for folder in _SEASON_FOLDERS
    }
    return {"status": "ok", "season": season, "downloaded": downloaded}


def _execute_retrain_job(job_id: str) -> None:
    session = SessionLocal()
    try:
        run_production_training(session)
        _retrain_jobs[job_id] = {"status": "done", "error": None}
    except Exception as exc:  # noqa: BLE001 - job failure is reported, not raised
        _retrain_jobs[job_id] = {"status": "failed", "error": str(exc)}
    finally:
        session.close()


@router.post("/retrain")
def start_retrain(background_tasks: BackgroundTasks) -> dict:
    """Kick off a production retrain as a background task (PRD §14).

    Returns a job id immediately - retraining is slow (~1 hour per §8.1's
    measured full-history runtime) and must never block a request. Does
    *not* refresh data first (§19 - two independent admin actions).
    """
    job_id = str(uuid.uuid4())
    _retrain_jobs[job_id] = {"status": "running", "error": None}
    background_tasks.add_task(_execute_retrain_job, job_id)
    return {"job_id": job_id, "status": "running"}


@router.get("/retrain/{job_id}")
def get_retrain_status(job_id: str) -> dict:
    job = _retrain_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown retrain job_id {job_id!r}")
    return {"job_id": job_id, **job}


def _serialize_model_version(record: ModelVersionRecord) -> dict:
    return {
        "id": record.id,
        "artifact_path": record.artifact_path,
        "through_date": record.through_date.isoformat(),
        "created_at": record.created_at.isoformat(),
        "is_active": record.is_active,
    }


@router.get("/models")
def list_models(session: Session = Depends(get_session)) -> list[dict]:
    repository = build_model_version_repository(session)
    return [_serialize_model_version(record) for record in repository.list()]


@router.post("/models/{version_id}/activate")
def activate_model(version_id: int, session: Session = Depends(get_session)) -> dict:
    """Roll back/switch the active model (PRD §14) - no retraining needed."""
    repository = build_model_version_repository(session)
    record = repository.activate(version_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown model version id {version_id}")
    return _serialize_model_version(record)


class RosterOverrideIn(BaseModel):
    player_id: int
    player_name: str
    team_id: int
    status: str
    new_team_id: int | None = None
    note: str | None = None


def _serialize_roster_override(record: RosterOverrideRecord) -> dict:
    return {
        "id": record.id,
        "player_id": record.player_id,
        "player_name": record.player_name,
        "team_id": record.team_id,
        "status": record.status,
        "new_team_id": record.new_team_id,
        "note": record.note,
    }


@router.get("/roster-overrides")
def list_roster_overrides(session: Session = Depends(get_session)) -> list[dict]:
    repository = build_roster_override_repository(session)
    return [_serialize_roster_override(record) for record in repository.list()]


@router.post("/roster-overrides")
def upsert_roster_override(
    payload: RosterOverrideIn, session: Session = Depends(get_session)
) -> dict:
    """Add a new override, or update the existing one for `payload.player_id` (§15).

    A single endpoint for both cases, matched to the "current state, no
    history" data model - there's only ever one row per player.
    """
    repository = build_roster_override_repository(session)
    existing = repository.get_by_player_id(payload.player_id)
    try:
        if existing is not None:
            record = repository.update(existing.id, **payload.model_dump(exclude={"player_id"}))
        else:
            record = repository.add(**payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _serialize_roster_override(record)


@router.delete("/roster-overrides/{override_id}")
def delete_roster_override(override_id: int, session: Session = Depends(get_session)) -> dict:
    repository = build_roster_override_repository(session)
    if not repository.remove(override_id):
        raise HTTPException(status_code=404, detail=f"Unknown roster override id {override_id}")
    return {"status": "ok"}


def _serialize_prediction(record: PredictionRecord) -> dict:
    return {
        "id": record.id,
        "game_id": record.game_id,
        "date": record.date,
        "home_team_id": record.home_team_id,
        "home_team_name": record.home_team_name,
        "away_team_id": record.away_team_id,
        "away_team_name": record.away_team_name,
        "predicted_margin": record.predicted_margin,
        "predicted_winner": record.predicted_winner,
        "home_win_probability": record.home_win_probability,
        "model_version_id": record.model_version_id,
        "captured_at": record.captured_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
        "actual_margin": record.actual_margin,
        "actual_winner": record.actual_winner,
        "reconciled_at": record.reconciled_at.isoformat() if record.reconciled_at else None,
    }


@router.get("/predictions")
def list_predictions(session: Session = Depends(get_session)) -> list[dict]:
    repository = build_prediction_repository(session)
    return [_serialize_prediction(record) for record in repository.list()]


@router.post("/predictions/capture")
def capture_predictions(
    horizon_days: int = Query(14, ge=1, le=60, description="How many days ahead to capture."),
    session: Session = Depends(get_session),
) -> list[dict]:
    """Insert/refresh a prediction row for every not-yet-played game in the next `horizon_days`.

    Never overwrites an already-reconciled row (§ live tracking design).
    The default of 14 keeps the feature-matrix build tractable and the
    forecast meaningful - see `prediction.predict_pending_games`'s
    docstring before pushing this much higher.
    """
    try:
        records = capture_pending_predictions(session, horizon_days=horizon_days)
    except NoActiveModelError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return [_serialize_prediction(record) for record in records]


@router.post("/predictions/reconcile")
def reconcile_predictions_endpoint(session: Session = Depends(get_session)) -> list[dict]:
    """Fill in actual results for pending predictions whose game is now complete locally.

    Reads only the locally cached schedule data - run `refresh-data` first
    to pick up games that finished since the last pull.
    """
    records = reconcile_predictions(session)
    return [_serialize_prediction(record) for record in records]


@router.get("/predictions/accuracy")
def get_prediction_accuracy(session: Session = Depends(get_session)) -> dict:
    return compute_accuracy(session)
