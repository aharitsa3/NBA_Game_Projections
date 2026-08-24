"""Current-state roster-override table and its data-access layer.

Per PRD §15, this replaces Phase 1's `prediction_date`-scoped
`config/roster_overrides.json` (T2.4) with a SQLite table holding
"what's true right now" - one row per player, in effect until an admin
changes or removes it. T7.2 adapts rows from this table into the
`RosterOverride` shape `features/roster_overrides.py`'s
`apply_roster_overrides()` already expects; this module only owns
persistence, no HTTP (that's T7.1).
"""

from __future__ import annotations

from sqlalchemy.orm import Mapped, Session, mapped_column

from nba_game_projections.app.db import Base
from nba_game_projections.features.roster_overrides import _VALID_STATUSES


class RosterOverrideRecord(Base):
    __tablename__ = "roster_overrides"

    id: Mapped[int] = mapped_column(primary_key=True)
    player_id: Mapped[int] = mapped_column(unique=True, index=True)
    player_name: Mapped[str]
    team_id: Mapped[int]
    status: Mapped[str]
    new_team_id: Mapped[int | None] = mapped_column(default=None)
    note: Mapped[str | None] = mapped_column(default=None)


def _validate_status(status: str) -> None:
    if status not in _VALID_STATUSES:
        raise ValueError(
            f"Unknown roster override status {status!r}; expected one of "
            f"{sorted(_VALID_STATUSES)}"
        )


class RosterOverrideRepository:
    """CRUD access to the current-state roster-override table."""

    def __init__(self, session: Session):
        self._session = session

    def list(self) -> list[RosterOverrideRecord]:
        return list(self._session.query(RosterOverrideRecord).order_by(RosterOverrideRecord.id))

    def get_by_player_id(self, player_id: int) -> RosterOverrideRecord | None:
        return (
            self._session.query(RosterOverrideRecord)
            .filter(RosterOverrideRecord.player_id == player_id)
            .one_or_none()
        )

    def add(
        self,
        *,
        player_id: int,
        player_name: str,
        team_id: int,
        status: str,
        new_team_id: int | None = None,
        note: str | None = None,
    ) -> RosterOverrideRecord:
        _validate_status(status)
        record = RosterOverrideRecord(
            player_id=player_id,
            player_name=player_name,
            team_id=team_id,
            status=status,
            new_team_id=new_team_id,
            note=note,
        )
        self._session.add(record)
        self._session.commit()
        self._session.refresh(record)
        return record

    def update(self, override_id: int, **fields: object) -> RosterOverrideRecord | None:
        record = self._session.get(RosterOverrideRecord, override_id)
        if record is None:
            return None
        if "status" in fields and fields["status"] is not None:
            _validate_status(fields["status"])
        for field, value in fields.items():
            setattr(record, field, value)
        self._session.commit()
        self._session.refresh(record)
        return record

    def remove(self, override_id: int) -> bool:
        record = self._session.get(RosterOverrideRecord, override_id)
        if record is None:
            return False
        self._session.delete(record)
        self._session.commit()
        return True


def build_roster_override_repository(session: Session) -> RosterOverrideRepository:
    return RosterOverrideRepository(session)
