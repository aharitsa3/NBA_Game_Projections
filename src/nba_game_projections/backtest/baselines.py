"""Reporting baselines to compare the trained model against (PRD §8).

Unlike the walk-forward harness (T4.1), none of these need their own
train/test fold-splitting logic: each is inherently causal by construction
(sequential dependence only on strictly-earlier games), computed once over
the full schedule history passed in, exactly like `features/context.py` and
`features/elo.py` already are. Whoever builds T4.3 filters/joins these
predictions down to whichever test-fold `game_id`s are in play.

Every function returns a `pd.Series` indexed by `game_id`, named
`predicted_home_win`, using pandas' nullable boolean dtype (`"boolean"`) so
a baseline can represent "no pick available" for a game via `pd.NA`. Only
`vegas_favorite` ever does this (a pick'em line, or a game with no matching
betting-lines row) — the other three baselines are always fully populated.
"""

from __future__ import annotations

import pandas as pd

from nba_game_projections.features.elo import compute_elo_ratings, elo_win_probability

# schedules-side abbreviation -> betting_lines-side abbreviation, for the six
# teams the two hoopR/betting_lines sources spell differently. Identity for
# every other team. Confirmed empirically against the real 2019-2021 archive.
_ABBREV_TO_BETTING_LINES = {
    "GS": "GSW",
    "NO": "NOP",
    "NY": "NYK",
    "SA": "SAS",
    "UTAH": "UTA",
    "WSH": "WAS",
}


def always_home(schedules: pd.DataFrame) -> pd.Series:
    """Baseline #1: always pick the home team."""
    return pd.Series(
        True,
        index=pd.Index(schedules["game_id"], name="game_id"),
        dtype="boolean",
        name="predicted_home_win",
    )


def better_record(schedules: pd.DataFrame) -> pd.Series:
    """Baseline #2: pick the team with the better win pct entering the game.

    Win percentage (not raw win count) computed from strictly-prior games
    within the same season only (records don't carry over across seasons).
    A team's first game of a season has no record yet, treated as a neutral
    0.5 rather than 0.0 or NaN. Ties (including the opening-day 0.5-vs-0.5
    case) default to picking the home team, same convention as baseline #1.
    """
    base_columns = ["game_id", "date", "season"]
    home = schedules[[*base_columns, "home_team_id", "home_winner"]].rename(
        columns={"home_team_id": "team_id", "home_winner": "won"}
    )
    away = schedules[[*base_columns, "away_team_id", "away_winner"]].rename(
        columns={"away_team_id": "team_id", "away_winner": "won"}
    )
    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["team_id", "season", "date"], kind="stable").reset_index(drop=True)

    grouped = long.groupby(["team_id", "season"], sort=False)
    games_played = grouped.cumcount()
    prior_wins = grouped["won"].cumsum() - long["won"].astype(int)

    win_pct = pd.Series(0.5, index=long.index)
    has_history = games_played > 0
    win_pct[has_history] = prior_wins[has_history] / games_played[has_history]
    long["win_pct"] = win_pct

    # Re-derive home/away win_pct per game by joining back to `schedules`,
    # rather than relying on the concat order, since sorting by
    # (team_id, season, date) above has already interleaved home/away rows.
    home_pct = long.merge(schedules[["game_id", "home_team_id"]], on="game_id")
    home_pct = home_pct[home_pct["team_id"] == home_pct["home_team_id"]][["game_id", "win_pct"]]
    home_pct = home_pct.rename(columns={"win_pct": "home_win_pct"})

    away_pct = long.merge(schedules[["game_id", "away_team_id"]], on="game_id")
    away_pct = away_pct[away_pct["team_id"] == away_pct["away_team_id"]][["game_id", "win_pct"]]
    away_pct = away_pct.rename(columns={"win_pct": "away_win_pct"})

    merged = schedules[["game_id"]].merge(home_pct, on="game_id").merge(away_pct, on="game_id")
    predicted = (merged["home_win_pct"] >= merged["away_win_pct"]).astype("boolean")
    predicted.index = pd.Index(merged["game_id"], name="game_id")

    return predicted.rename("predicted_home_win")


def elo_alone(
    schedules: pd.DataFrame,
    k_factor: float = 20.0,
    initial_rating: float = 1500.0,
    home_advantage: float = 100.0,
) -> pd.Series:
    """Baseline #3: pick per Elo rating alone (isolates its standalone signal).

    `compute_elo_ratings`'s `pre_game_rating` does NOT have home-court
    advantage baked in — `home_advantage` is applied only transiently inside
    `elo_win_probability`'s expected-score formula, never persisted into the
    stored rating. Comparing the two raw ratings directly would silently
    drop home-court advantage from this baseline, so the win probability
    (which does apply it) is what's compared against 0.5, not the ratings.
    """
    ratings = compute_elo_ratings(
        schedules, k_factor=k_factor, initial_rating=initial_rating, home_advantage=home_advantage
    )
    home_ratings = ratings.loc[ratings["is_home"], ["game_id", "pre_game_rating"]].rename(
        columns={"pre_game_rating": "home_rating"}
    )
    away_ratings = ratings.loc[~ratings["is_home"], ["game_id", "pre_game_rating"]].rename(
        columns={"pre_game_rating": "away_rating"}
    )
    merged = home_ratings.merge(away_ratings, on="game_id")

    win_prob = elo_win_probability(merged["home_rating"], merged["away_rating"], home_advantage)
    predicted = (win_prob > 0.5).astype("boolean")
    predicted.index = pd.Index(merged["game_id"], name="game_id")

    return predicted.rename("predicted_home_win")


def vegas_favorite(schedules: pd.DataFrame, betting_lines: pd.DataFrame) -> pd.Series:
    """Baseline #4: pick per the Vegas closing line (reference only, PRD §8).

    `betting_lines` has no shared `game_id` with `schedules`, so the join
    key is `(game_date, home_abbrev, away_abbrev)` instead — deliberately
    *not* `season`: `betting_lines.season` uses the season-start-year
    convention (e.g. 2018 for the 2018-19 season) while `schedules.season`
    uses season-end-year (2019 for the same season, see
    `config.current_season()`), so joining on season (even shifted) drops
    every row. A handful of real games (~1.3%, mostly the 2020 COVID-bubble
    restart) have no matching betting_lines row at all and simply produce no
    prediction (`pd.NA`), which is an upstream archive gap, not a bug here.

    `line` is signed relative to the home team (negative = home favored,
    per `betting_lines.py`'s docstring). `line == 0` is a pick'em with no
    established favorite, so it returns `pd.NA` rather than a guess, same as
    an unmatched game.
    """
    sched = schedules[
        ["game_id", "date", "home_team_abbreviation", "away_team_abbreviation"]
    ].copy()
    sched["game_date"] = sched["date"].dt.tz_localize(None).dt.normalize()
    sched["home_abbrev"] = sched["home_team_abbreviation"].replace(_ABBREV_TO_BETTING_LINES)
    sched["away_abbrev"] = sched["away_team_abbreviation"].replace(_ABBREV_TO_BETTING_LINES)

    lines = betting_lines[["game_date", "home_team_abbrev", "visit_team_abbrev", "line"]].copy()
    lines["game_date"] = lines["game_date"].dt.normalize()

    merged = sched.merge(
        lines,
        left_on=["game_date", "home_abbrev", "away_abbrev"],
        right_on=["game_date", "home_team_abbrev", "visit_team_abbrev"],
        how="left",
    )

    predicted = pd.Series(pd.NA, index=merged.index, dtype="boolean")
    predicted[merged["line"] < 0] = True
    predicted[merged["line"] > 0] = False
    predicted.index = pd.Index(merged["game_id"], name="game_id")

    return predicted.rename("predicted_home_win")
