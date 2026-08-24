"""Versioned persistence for trained models (PRD §14).

Each production train (T6.1) writes a new `joblib` artifact under
`config.MODEL_ARTIFACTS_DIR` without deleting prior versions, and this
repository tracks which one is currently "active" - a bad retrain is
recoverable by flipping the active pointer back, without retraining.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import joblib
from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, Session, mapped_column

from nba_game_projections import config
from nba_game_projections.app.db import Base
from nba_game_projections.models.train import TrainedModel


class ModelVersionRecord(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    artifact_path: Mapped[str] = mapped_column(String)
    through_date: Mapped[datetime] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    is_active: Mapped[bool] = mapped_column(default=False)


class ModelVersionRepository:
    """Save/activate/load trained-model artifacts and their metadata."""

    def __init__(self, session: Session):
        self._session = session

    def save(self, trained_model: TrainedModel) -> ModelVersionRecord:
        config.MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        through_date = trained_model.through_date.to_pydatetime()
        artifact_name = f"model_{through_date:%Y%m%d}_{uuid.uuid4().hex[:8]}.joblib"
        artifact_path = config.MODEL_ARTIFACTS_DIR / artifact_name
        joblib.dump(trained_model, artifact_path)

        record = ModelVersionRecord(
            artifact_path=str(artifact_path), through_date=through_date, is_active=False
        )
        self._session.add(record)
        self._session.commit()
        self._session.refresh(record)
        return record

    def list(self) -> list[ModelVersionRecord]:
        return list(self._session.query(ModelVersionRecord).order_by(ModelVersionRecord.id))

    def get_active(self) -> ModelVersionRecord | None:
        return (
            self._session.query(ModelVersionRecord)
            .filter(ModelVersionRecord.is_active.is_(True))
            .one_or_none()
        )

    def activate(self, version_id: int) -> ModelVersionRecord | None:
        record = self._session.get(ModelVersionRecord, version_id)
        if record is None:
            return None
        self._session.query(ModelVersionRecord).filter(
            ModelVersionRecord.is_active.is_(True)
        ).update({"is_active": False})
        record.is_active = True
        self._session.commit()
        self._session.refresh(record)
        return record

    def load_active_model(self) -> TrainedModel | None:
        active = self.get_active()
        if active is None:
            return None
        return joblib.load(active.artifact_path)


def build_model_version_repository(session: Session) -> ModelVersionRepository:
    return ModelVersionRepository(session)
