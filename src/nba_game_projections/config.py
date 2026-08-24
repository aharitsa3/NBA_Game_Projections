"""Central configuration for the NBA Game Projections pipeline.

All data directories, the hoopR source URL, the season range, and the
roster-override file path live here. No other module should hardcode a
path or URL — import from this module instead.
"""

from __future__ import annotations

import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

CONFIG_DIR = PROJECT_ROOT / "config"
ROSTER_OVERRIDES_PATH = CONFIG_DIR / "roster_overrides.json"

# Phase 2: app-managed state (roster overrides, model version metadata).
# Gitignored mutable runtime state, same treatment as data/raw/ and
# config/roster_overrides.json (PRD §14).
APP_DB_PATH = DATA_PROCESSED_DIR / "app.db"
APP_DATABASE_URL = f"sqlite:///{APP_DB_PATH}"
MODEL_ARTIFACTS_DIR = DATA_PROCESSED_DIR / "models"

HOOPR_BASE_URL = "https://raw.githubusercontent.com/sportsdataverse/hoopR-nba-data/main"
HOOPR_FOLDERS = ("player_box", "team_box", "schedules", "betting_lines")

# Phase 1 historical window: betting_lines (the reporting baseline) only
# goes back to 2017, so every training/test season stays comparable
# against a market baseline (PRD §4.1).
SEASON_START = 2017


def current_season() -> int:
    """Return the current hoopR season label (the season's ending year).

    hoopR/ESPN name a season by the calendar year its second half falls
    in (e.g. the 2023-24 season is "2024"). The NBA season starts in
    October, so a game played October-December belongs to the *next*
    calendar year's season label.
    """
    today = datetime.date.today()
    return today.year + 1 if today.month >= 10 else today.year


def season_range() -> range:
    """Seasons in scope for Phase 1: SEASON_START through the current season, inclusive."""
    return range(SEASON_START, current_season() + 1)
