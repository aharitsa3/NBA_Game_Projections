"""Adapts the live roster-override table (T5.2) into T2.4's `RosterOverride` shape.

Per PRD §15, only the *source* of the override list changes for Phase 2 -
SQLite's current state instead of a `prediction_date`-scoped JSON file -
not how it's applied to a computed roster. `apply_roster_overrides()`
(`features/roster_overrides.py`, unchanged) is reused as-is against
whatever this adapter returns.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from nba_game_projections.app.repositories.roster_overrides import (
    build_roster_override_repository,
)
from nba_game_projections.features.roster_overrides import RosterOverride


def load_live_roster_overrides(session: Session) -> list[RosterOverride]:
    """Return every current override as a `RosterOverride`, read live from SQLite."""
    repository = build_roster_override_repository(session)
    return [
        RosterOverride(
            player_id=record.player_id,
            player_name=record.player_name,
            team_id=record.team_id,
            status=record.status,
            new_team_id=record.new_team_id,
            note=record.note,
        )
        for record in repository.list()
    ]
