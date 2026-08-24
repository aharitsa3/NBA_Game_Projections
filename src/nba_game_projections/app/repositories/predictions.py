"""Live prediction tracking: one row per game, locked once reconciled.

A row starts out as a mutable *forecast* - `capture`'s upsert can overwrite
its predicted values (e.g. after a retrain, or a roster-override edit)
for as long as the game hasn't been played. Once `reconcile` fills in the
actual result, the row becomes a permanent historical record and
`upsert_prediction` refuses to touch it again - only a reconciled row
means anything for accuracy measurement, so only a reconciled row needs
protecting from being overwritten.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, Session, mapped_column

from nba_game_projections.app.db import Base


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    game_id: Mapped[int] = mapped_column(unique=True, index=True)
    date: Mapped[str] = mapped_column(String)
    home_team_id: Mapped[int]
    home_team_name: Mapped[str | None] = mapped_column(default=None)
    away_team_id: Mapped[int]
    away_team_name: Mapped[str | None] = mapped_column(default=None)
    predicted_margin: Mapped[float]
    predicted_winner: Mapped[str]
    home_win_probability: Mapped[float]
    model_version_id: Mapped[int]
    captured_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    actual_margin: Mapped[float | None] = mapped_column(default=None)
    actual_winner: Mapped[str | None] = mapped_column(default=None)
    reconciled_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)


class PredictionRepository:
    """Upsert-for-pending / lock-on-reconcile access to the predictions table."""

    def __init__(self, session: Session):
        self._session = session

    def list(self) -> list[PredictionRecord]:
        return list(self._session.query(PredictionRecord).order_by(PredictionRecord.id))

    def list_pending(self) -> list[PredictionRecord]:
        return list(
            self._session.query(PredictionRecord)
            .filter(PredictionRecord.reconciled_at.is_(None))
            .order_by(PredictionRecord.id)
        )

    def get_by_game_id(self, game_id: int) -> PredictionRecord | None:
        return (
            self._session.query(PredictionRecord)
            .filter(PredictionRecord.game_id == game_id)
            .one_or_none()
        )

    def upsert_prediction(
        self,
        *,
        game_id: int,
        date: str,
        home_team_id: int,
        home_team_name: str | None,
        away_team_id: int,
        away_team_name: str | None,
        predicted_margin: float,
        predicted_winner: str,
        home_win_probability: float,
        model_version_id: int,
    ) -> PredictionRecord:
        """Insert a new row, or refresh an existing *unreconciled* row's predicted values.

        A reconciled row is returned unchanged - it's the historical
        accuracy record and must never be overwritten by a later capture.
        """
        existing = self.get_by_game_id(game_id)
        if existing is not None:
            if existing.reconciled_at is not None:
                return existing
            existing.date = date
            existing.home_team_id = home_team_id
            existing.home_team_name = home_team_name
            existing.away_team_id = away_team_id
            existing.away_team_name = away_team_name
            existing.predicted_margin = predicted_margin
            existing.predicted_winner = predicted_winner
            existing.home_win_probability = home_win_probability
            existing.model_version_id = model_version_id
            existing.updated_at = datetime.now(UTC)
            self._session.commit()
            self._session.refresh(existing)
            return existing

        record = PredictionRecord(
            game_id=game_id,
            date=date,
            home_team_id=home_team_id,
            home_team_name=home_team_name,
            away_team_id=away_team_id,
            away_team_name=away_team_name,
            predicted_margin=predicted_margin,
            predicted_winner=predicted_winner,
            home_win_probability=home_win_probability,
            model_version_id=model_version_id,
        )
        self._session.add(record)
        self._session.commit()
        self._session.refresh(record)
        return record

    def reconcile(
        self, game_id: int, *, actual_margin: float, actual_winner: str
    ) -> PredictionRecord | None:
        """Fill in the actual result and lock the row. No-op if unknown or already locked."""
        record = self.get_by_game_id(game_id)
        if record is None or record.reconciled_at is not None:
            return None
        record.actual_margin = actual_margin
        record.actual_winner = actual_winner
        record.reconciled_at = datetime.now(UTC)
        self._session.commit()
        self._session.refresh(record)
        return record


def build_prediction_repository(session: Session) -> PredictionRepository:
    return PredictionRepository(session)
