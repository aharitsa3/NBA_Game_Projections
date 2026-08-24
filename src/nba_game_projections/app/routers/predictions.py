"""Prediction-serving endpoint (T8.1, PRD §17)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from nba_game_projections.app.db import get_session
from nba_game_projections.app.services.prediction import NoActiveModelError, predict_slate

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("/slate")
def get_slate(date: str | None = None, session: Session = Depends(get_session)) -> list[dict]:
    """Every game scheduled on `date` (defaults to today) plus its prediction."""
    try:
        return predict_slate(session, date=date)
    except NoActiveModelError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
