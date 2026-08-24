---
name: refresh-predictions
description: Refreshes this project's hoopR game data and updates NBA game predictions without needing the FastAPI server, Swagger UI, or the browser UI running. Use this whenever the user asks to refresh data, pull the latest game results, update or regenerate predictions, sync predictions with actual results, or run the daily/routine data refresh for the NBA Game Projections project - even if they don't explicitly name the API endpoints or say "refresh-data" / "capture" / "reconcile". Always prefer this skill over manually walking the user through curl commands or the Swagger docs for this project.
---

# Refresh Predictions

Runs the same three admin actions the app exposes via its UI/Swagger docs
(`/admin/refresh-data`, `/admin/predictions/capture`,
`/admin/predictions/reconcile`), but as one direct script against the
SQLite database — no `uvicorn` server needs to be running.

## What it does, in order

1. **Refresh data** — downloads any not-yet-cached hoopR files (player box
   scores, team box scores, schedules) for the current season.
2. **Capture predictions** — inserts/refreshes a prediction for every
   not-yet-played game in the next N days (default 14), using whichever
   model version is currently active. Never overwrites a prediction whose
   game has already been reconciled.
3. **Reconcile** — fills in actual results for any pending prediction
   whose game is now complete in the just-refreshed data.

## Running it

```bash
uv run python .claude/skills/refresh-predictions/scripts/refresh.py
```

Run this from the project root (`NBA_Game_Projections/`) so `uv run`
resolves the project's environment. Pass `--horizon-days N` to change how
far ahead predictions get captured (default 14, matching the API's own
default and cap of 60 — see `app/services/prediction.py` for why that cap
exists before raising it much higher).

## Interpreting the output

The script prints a plain-text summary of each step — relay it to the
user as-is: counts of files downloaded, predictions captured/refreshed,
and predictions reconciled.

If it reports **no active model version**, nobody has trained and
activated a model yet. Tell the user to do that first (`POST
/admin/retrain` via the UI's Admin tab, or Swagger — it's a background job
that can take ~an hour), then re-run this skill. Don't try to train a
model yourself as part of this skill; retraining is a separate, slow
operation with its own concerns.
