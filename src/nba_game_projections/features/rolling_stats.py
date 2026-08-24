"""Roster-aware trailing rolling stats, aggregated to a team-level feature vector.

A team's roster is derived from the team's own trailing games rather than
tracked explicitly: `player_box["team_id"]` already reflects trades on a
per-game basis, so scanning team T's own most recent games for who appears
under `team_id == T` naturally phases a traded-away player out (as older
pre-trade games roll off the window) and phases a traded-in player in (as
post-trade games with team T accumulate) — no separate trade-tracking logic
required (PRD §5/§6).
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

_COUNTING_STATS = ("points", "rebounds", "assists", "steals", "blocks", "turnovers")


def team_roster(
    player_box: pd.DataFrame, team_id: int, as_of_date: pd.Timestamp, window: int
) -> list[int]:
    """The automatic trailing-active-roster default: distinct players who appear
    in team `team_id`'s own last `window` games strictly before `as_of_date`.

    Public so callers (e.g. feature assembly) can compute this roster, run it
    through T2.4's override deltas, and pass the adjusted set back in via
    `team_rolling_features`'s `active_player_ids` parameter.
    """
    team_rows = player_box[
        (player_box["team_id"] == team_id) & (player_box["game_date"] < as_of_date)
    ]
    if team_rows.empty:
        return []

    recent_game_ids = (
        team_rows[["game_id", "game_date"]]
        .drop_duplicates("game_id")
        .sort_values("game_date", ascending=False)
        .head(window)["game_id"]
    )
    roster_rows = team_rows[team_rows["game_id"].isin(recent_game_ids)]
    return list(pd.unique(roster_rows["athlete_id"]))


def _player_trailing_stats(
    player_box: pd.DataFrame, athlete_id: int, as_of_date: pd.Timestamp, window: int
) -> dict[str, float]:
    """Trailing per-game averages from a player's own last `window` games played.

    Deliberately not bounded to any one team's games: a just-traded or
    just-returned-from-injury player's own recent history may span multiple
    teams, and this is what T2.1's aggregation step relies on.
    """
    rows = player_box[
        (player_box["athlete_id"] == athlete_id)
        & (player_box["game_date"] < as_of_date)
        & (~player_box["did_not_play"])
    ]
    rows = rows.drop_duplicates("game_id").sort_values("game_date", ascending=False).head(window)

    if rows.empty:
        return {
            "games_played": 0,
            "minutes": 0.0,
            "field_goals_made": 0.0,
            "field_goals_attempted": 0.0,
            "three_point_field_goals_made": 0.0,
            **dict.fromkeys(_COUNTING_STATS, 0.0),
        }

    stats = {stat: float(rows[stat].mean()) for stat in _COUNTING_STATS}
    stats["games_played"] = len(rows)
    stats["minutes"] = float(rows["minutes"].mean())
    stats["field_goals_made"] = float(rows["field_goals_made"].sum())
    stats["field_goals_attempted"] = float(rows["field_goals_attempted"].sum())
    stats["three_point_field_goals_made"] = float(rows["three_point_field_goals_made"].sum())
    return stats


def team_rolling_features(
    player_box: pd.DataFrame,
    team_id: int,
    as_of_date,
    window: int = 10,
    active_player_ids: Iterable[int] | None = None,
) -> dict[str, float]:
    """Trailing-`window`-game team feature vector as of `as_of_date` (exclusive).

    Roster is either the automatic trailing-active-roster default (derived
    from team `team_id`'s own last `window` games strictly before
    `as_of_date`) or, when `active_player_ids` is given, that explicit
    player set instead (the hook T2.4's manual override output uses).
    """
    as_of_date = pd.Timestamp(as_of_date)

    roster_ids = (
        list(dict.fromkeys(active_player_ids))
        if active_player_ids is not None
        else team_roster(player_box, team_id, as_of_date, window)
    )

    player_stats = [
        _player_trailing_stats(player_box, athlete_id, as_of_date, window)
        for athlete_id in roster_ids
    ]

    totals = {stat: sum(s[stat] for s in player_stats) for stat in _COUNTING_STATS}
    minutes_per_game = sum(s["minutes"] for s in player_stats)

    weighted_efg_sum = 0.0
    efg_weight_total = 0.0
    for s in player_stats:
        fga = s["field_goals_attempted"]
        if fga > 0 and minutes_per_game > 0:
            weight = s["minutes"] / minutes_per_game
            player_efg = (s["field_goals_made"] + 0.5 * s["three_point_field_goals_made"]) / fga
            weighted_efg_sum += player_efg * weight
            efg_weight_total += weight
    efg_pct = weighted_efg_sum / efg_weight_total if efg_weight_total > 0 else float("nan")

    return {
        "points": totals["points"],
        "rebounds": totals["rebounds"],
        "assists": totals["assists"],
        "steals": totals["steals"],
        "blocks": totals["blocks"],
        "turnovers": totals["turnovers"],
        "efg_pct": efg_pct,
        "minutes_per_game": minutes_per_game,
        "roster_size": len(roster_ids),
    }
