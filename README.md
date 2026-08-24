# NBA Game Projections

Predicts NBA game outcomes: a LightGBM regression model forecasts the margin of victory for
a matchup, and a win probability is derived from that margin plus the model's historical
residual distribution. Trained on player box scores, team box scores, schedules, and betting
lines dating back to 2017 (via [hoopR](https://github.com/sportsdataverse/hoopR-nba-data)),
using roster-aware rolling stats, rest-day/back-to-back context, and Elo ratings as features.

Walk-forward backtesting (never k-fold, to avoid leakage through rolling features) shows the
model beating simple baselines (always-pick-home, better-record, Elo-alone) on win/loss
accuracy, while trailing the closing Vegas line — used only as a reference, never as an
optimization target.

The project has two parts:

- **Prediction pipeline** (`data/`, `features/`, `models/`, `backtest/`) — the offline ML
  pipeline: downloads and caches hoopR data, builds features, trains the model, and validates
  it via walk-forward backtesting.
- **Serving app** (`app/`) — a FastAPI + SQLite application built on top of the pipeline:
  serves live predictions, manages roster overrides and model versions, supports
  admin-triggered retraining, and tracks predictions against actual results over time so
  real-world accuracy can be measured.

## Project structure

```
NBA_Game_Projections/
├── config/
│   └── roster_overrides.example.json   # example schema for the gitignored override file
├── scripts/
│   └── run_backtest.py                 # CLI: full walk-forward backtest + report
├── src/
│   └── nba_game_projections/
│       ├── config.py                   # paths, URLs, season range - single source of truth
│       │
│       ├── data/                       # download & load raw hoopR data
│       │   ├── download.py
│       │   ├── schedules.py
│       │   ├── upcoming_games.py
│       │   ├── player_box.py
│       │   ├── team_box.py
│       │   └── betting_lines.py
│       │
│       ├── features/                   # feature engineering
│       │   ├── build.py                # assembles the full feature matrix
│       │   ├── rolling_stats.py        # roster-aware trailing per-player stats
│       │   ├── context.py              # rest days / back-to-back / home-away
│       │   ├── elo.py                  # Elo rating engine
│       │   └── roster_overrides.py     # file-based override application
│       │
│       ├── models/                     # training & inference
│       │   ├── train.py                # train(), wide_game_matrix()
│       │   └── win_probability.py      # margin -> win probability
│       │
│       ├── backtest/                   # walk-forward validation
│       │   ├── walk_forward.py
│       │   ├── baselines.py
│       │   └── report.py
│       │
│       └── app/                        # FastAPI serving application
│           ├── main.py                 # app instance, lifespan, static UI mount
│           ├── db.py                   # SQLAlchemy engine/session
│           ├── routers/
│           │   ├── predictions.py      # GET /predictions/slate
│           │   └── admin.py            # data refresh, retrain, models, overrides, tracking
│           ├── services/
│           │   ├── prediction.py       # slate / pending-game prediction logic
│           │   ├── prediction_tracking.py  # capture, reconcile, accuracy scoring
│           │   ├── training.py         # production retrain entrypoint
│           │   └── roster_overrides.py
│           ├── repositories/
│           │   ├── model_versions.py
│           │   ├── roster_overrides.py
│           │   └── predictions.py
│           └── static/
│               └── index.html          # built-in single-page UI
│
├── tests/                              # pytest, ~1:1 with src/ modules
├── PRD.md                              # product requirements / design record
├── tasks.md                            # implementation task breakdown
└── pyproject.toml
```

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies (including dev tools: pytest, ruff, httpx)
uv sync

# Download and cache hoopR data (player box scores, team box scores, schedules,
# betting lines) for every season from 2017 onward. Can take a while on first run.
uv run python -m nba_game_projections.data.download
```

## Running the app

```bash
uv run uvicorn nba_game_projections.app.main:app --reload
```

- **UI**: [http://127.0.0.1:8000](http://127.0.0.1:8000) — a lightweight built-in page for
  viewing the predicted slate, managing roster overrides, training/activating models, and
  reviewing live prediction accuracy.
- **Interactive API docs (Swagger)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

The app starts with no trained model. On first run: use **Admin → Start Retrain** in the UI
(or `POST /admin/retrain`) to train and activate a model before predictions will work.

Runs locally only — single-user, no authentication, no remote deployment.

## API reference

### Predictions

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/predictions/slate?date=YYYY-MM-DD` | Predicted margin, winner, and home win probability for every game on `date` (defaults to today). Computed fresh on every call — no caching. |

### Admin — data & models

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/admin/refresh-data` | Download any not-yet-cached hoopR files for the current season. Fast, synchronous. |
| `POST` | `/admin/retrain` | Kick off a full production retrain as a background task (can take ~an hour). Returns a `job_id` to poll. On success, the new model is saved, activated, and pending predictions are refreshed against it. |
| `GET` | `/admin/retrain/{job_id}` | Poll a retrain job's status (`running` / `done` / `failed`). |
| `GET` | `/admin/models` | List every trained model version and which one is active. |
| `POST` | `/admin/models/{version_id}/activate` | Activate a model version (e.g. to roll back a bad retrain), without retraining. |

### Admin — roster overrides

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/admin/roster-overrides` | List current roster overrides (injuries, trades) applied to every prediction. |
| `POST` | `/admin/roster-overrides` | Add a new override, or update the existing one for that `player_id`. |
| `DELETE` | `/admin/roster-overrides/{override_id}` | Remove an override. |

### Admin — prediction tracking

Live accuracy tracking, separate from the offline backtest: predictions are locked in
*before* a game is played and compared against the actual result afterward.

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/admin/predictions` | List every tracked prediction, pending and reconciled. |
| `POST` | `/admin/predictions/capture?horizon_days=14` | Insert/refresh a prediction for every not-yet-played game in the next `horizon_days` (default 14, max 60). Never overwrites an already-reconciled row. |
| `POST` | `/admin/predictions/reconcile` | Fill in actual results for pending predictions whose game is complete in the *locally cached* schedule data (run `refresh-data` first to pick up newly finished games). |
| `GET` | `/admin/predictions/accuracy` | Win/loss accuracy, margin MAE/RMSE, and Brier score over reconciled predictions — pooled and broken out per model version. |

### Health

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness check. |

## Development

```bash
uv run pytest                                   # run the test suite
uv run ruff check .                             # lint
uv run python scripts/run_backtest.py           # full walk-forward backtest + report (slow)
```
