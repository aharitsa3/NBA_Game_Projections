"""Parses `config/roster_overrides.json` and applies manual roster edits.

Per PRD §6.2, a roster override file holds one prediction date's worth of
manual in/out/reassignment edits to the automatically-computed trailing
roster - e.g. a same-day injury not yet reflected in trailing rolling
stats, or a trade not yet reflected in `player_box` `team_id`. This module
only computes the *delta* implied by the override file for a given team;
applying that delta to an automatically-computed roster is the caller's
(T2.1's) job.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from nba_game_projections import config

_VALID_STATUSES = {"in", "out", "reassigned"}


@dataclass(frozen=True)
class RosterOverride:
    player_id: int
    player_name: str
    team_id: int
    status: str
    new_team_id: int | None = None
    note: str | None = None


def load_roster_overrides(path: Path | None = None) -> tuple[date, list[RosterOverride]]:
    """Parse the override file. Defaults to `config.ROSTER_OVERRIDES_PATH`.

    Unlike the Wave 1 data loaders, where a missing cache is an error state
    the user must fix by downloading, a missing `roster_overrides.json` is
    a normal, common state - it's gitignored and most runs won't have one -
    meaning "no overrides for this run", so this returns an empty overrides
    list (with today's date) instead of raising.
    """
    resolved_path = path if path is not None else config.ROSTER_OVERRIDES_PATH
    if not resolved_path.exists():
        return date.today(), []

    raw = json.loads(resolved_path.read_text())
    prediction_date = date.fromisoformat(raw["prediction_date"])

    overrides = []
    for entry in raw["overrides"]:
        status = entry["status"]
        if status not in _VALID_STATUSES:
            raise ValueError(
                f"Unknown roster override status {status!r} for player_id "
                f"{entry.get('player_id')!r}; expected one of {sorted(_VALID_STATUSES)}"
            )
        overrides.append(
            RosterOverride(
                player_id=entry["player_id"],
                player_name=entry["player_name"],
                team_id=entry["team_id"],
                status=status,
                new_team_id=entry.get("new_team_id"),
                note=entry.get("note"),
            )
        )

    return prediction_date, overrides


def roster_override_deltas(
    team_id: int, overrides: list[RosterOverride]
) -> tuple[set[int], set[int]]:
    """Return (player_ids_to_add, player_ids_to_remove) for `team_id`."""
    additions: set[int] = set()
    removals: set[int] = set()

    for override in overrides:
        if override.status == "out":
            if override.team_id == team_id:
                removals.add(override.player_id)
        elif override.status == "in":
            if override.team_id == team_id:
                additions.add(override.player_id)
        elif override.status == "reassigned":
            if override.team_id == team_id:
                removals.add(override.player_id)
            if override.new_team_id == team_id:
                additions.add(override.player_id)

    return additions, removals


def apply_roster_overrides(
    team_id: int, automatic_roster: Iterable[int], overrides: list[RosterOverride]
) -> set[int]:
    """Convenience wrapper: (set(automatic_roster) - removals) | additions."""
    additions, removals = roster_override_deltas(team_id, overrides)
    return (set(automatic_roster) - removals) | additions
