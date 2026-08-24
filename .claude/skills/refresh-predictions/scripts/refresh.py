#!/usr/bin/env python3
"""Refresh hoopR data, capture predictions for pending games, and reconcile completed ones.

Mirrors the app's `/admin/refresh-data`, `/admin/predictions/capture`, and
`/admin/predictions/reconcile` endpoints, run directly against the SQLite
database - no running `uvicorn` server required.
"""

from __future__ import annotations

import argparse
import sys

from nba_game_projections import config
from nba_game_projections.app.db import SessionLocal, init_db
from nba_game_projections.app.services.prediction import NoActiveModelError
from nba_game_projections.app.services.prediction_tracking import (
    capture_pending_predictions,
    reconcile_predictions,
)
from nba_game_projections.data import download

_SEASON_FOLDERS = ("player_box", "team_box", "schedules")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=14,
        help="How many days ahead to capture predictions for (default: 14, matches the API).",
    )
    args = parser.parse_args()

    init_db()

    season = config.current_season()
    print(f"Refreshing hoopR data for season {season}...")
    downloaded_counts = {
        folder: len(download.download_season_files(folder, [season])) for folder in _SEASON_FOLDERS
    }
    print(f"  done: {downloaded_counts}")

    session = SessionLocal()
    try:
        print(f"Capturing/refreshing predictions for the next {args.horizon_days} day(s)...")
        try:
            captured = capture_pending_predictions(session, horizon_days=args.horizon_days)
            print(f"  {len(captured)} prediction(s) captured/refreshed.")
        except NoActiveModelError:
            print(
                "  Skipped: no active model version. Train and activate a model first "
                "(POST /admin/retrain, or the UI's Admin tab)."
            )

        print("Reconciling completed games...")
        reconciled = reconcile_predictions(session)
        print(f"  {len(reconciled)} prediction(s) reconciled.")
    finally:
        session.close()

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
