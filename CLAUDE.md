# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

NBA game outcome projections: predicts margin of victory via LightGBM regression, derives win probability from margin + residual distribution. Two phases, both complete:

- **Phase 1** — pure ML/data pipeline (data ingestion, feature engineering, training, walk-forward backtesting). Sole data source is hoopR (`sportsdataverse/hoopR-nba-data`), fetched unauthenticated over HTTPS and cached locally.
- **Phase 2** — FastAPI + SQLite application layer on top of Phase 1: admin-triggered retraining, roster-override CRUD, versioned model artifacts, and a `GET /predictions/slate` endpoint. Single-user, no auth, local-only (no remote deployment).

`PRD.md` and `tasks.md` are the detailed historical design record for both phases (all tasks complete). Don't extend or reference them for new work — treat them as background reading, not a convention to continue.

## Commands

- Install deps: `uv sync`
- Run tests: `uv run pytest`
- Lint: `uv run ruff check .`
- Run dev server: `uvicorn nba_game_projections.app.main:app --reload`
- Run full walk-forward backtest: `uv run python scripts/run_backtest.py` (slow — well over an hour; requires `data/raw/` already downloaded)

Package manager is **uv**, not pip/poetry. Python 3.13 (see `.python-version`).

## Architecture

`src/nba_game_projections/`:
- `config.py` — single source of truth for paths, URLs, and season range (`SEASON_START = 2017`). No other module should hardcode a path or URL — import from here instead.
- `data/` — loaders for completed data (`download.py`, `schedules.py`, `player_box.py`, `team_box.py`, `betting_lines.py`) and for not-yet-played games (`upcoming_games.py`, deliberately separate — `schedules.py`'s contract is "every row is a completed game").
- `features/` — `rolling_stats.py` (roster-aware trailing per-player stats), `context.py` (rest days/back-to-back/home-away), `elo.py`, `roster_overrides.py` (Phase 1 file-based overrides), `build.py` (assembles the full feature matrix).
- `models/` — `train.py` (`train(through_date)` entrypoint; also exposes `wide_game_matrix`, the public home/away-pairing helper reused by both training and Phase 2 prediction), `win_probability.py`.
- `backtest/` — `walk_forward.py` (expanding-window driver), `baselines.py`, `report.py`.
- `app/` — FastAPI app: `main.py` (app instance, `init_db()` lifespan, static UI mount at `/`), `db.py`, `routers/` (`admin.py`, `predictions.py`), `services/` (`prediction.py`, `training.py`, `roster_overrides.py`), `repositories/` (SQLAlchemy models + CRUD, one per entity).

Tests in `tests/` are flat, roughly 1:1 with `src/` modules.

## Gotchas

- **No environment variables anywhere** — all config is hardcoded constants in `config.py`, not env-driven. Don't add `.env`/`os.getenv` config without discussing it first; it'd be a deliberate departure from the current pattern.
- **`features/rolling_stats.py` is slow** — it rescans the full `player_box` table per team-game row. A full 9-season backtest takes ~1 hour. Anything touching this module or writing new backtests should expect long runtimes; this is why retraining runs as a background task in the app rather than inline.
- **Data and DB are gitignored and reproducible, not source-controlled.** `data/raw/` and `data/processed/` (including SQLite `app.db` and joblib model artifacts) must be regenerated locally — via `data/download.py` or `POST /admin/refresh-data` — after a fresh clone. There's no migration tool (SQLite schema is created via `create_all` on startup).
- **FastAPI mutable-default handling**: `fastapi.Depends`, `Query`, and `Body` are marked `extend-immutable-calls` for ruff's flake8-bugbear (B008) rule in `pyproject.toml`. Preserve this when adding new endpoints that use similar FastAPI defaults.
- Predictions are computed fresh on every `/predictions/slate` request — no caching — so a roster-override edit is reflected on the very next request.
