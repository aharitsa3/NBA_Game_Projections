# NBA Game Projections — Implementation Tasks

**Phase 1** (Waves 0-4): derived from `PRD.md` §1-11 (pure ML/data pipeline,
no application layer). Complete - see `PRD.md` §8.1 for the actual backtest
results that gated Phase 2.

**Phase 2** (Waves 5-8): derived from `PRD.md` §12-20 (application - FastAPI
backend, SQLite-backed admin state, no frontend in this pass per §13). Scoped
2026-08-20 via structured interview. Complete - built 2026-08-22.

Tasks are grouped into dependency waves. Tasks within the same wave have no
dependencies on each other and can be built in parallel. Each task is single
scoped: one module (or tight cluster of files), with its own tests, so it can
be picked up and finished independently.

## Progress

- [x] Wave 0 — Project scaffolding (1 task)
- [x] Wave 1 — Data acquisition and loading (5 tasks)
- [x] Wave 2 — Feature engineering (5 tasks)
- [x] Wave 3 — Model (2 tasks)
- [x] Wave 4 — Backtest and reporting (3 tasks)
- [x] Wave 5 — App scaffolding, persistence, and independent prep (5 tasks)
- [x] Wave 6 — Model lifecycle and admin operations (4 tasks)
- [x] Wave 7 — Roster override admin API (2 tasks)
- [x] Wave 8 — Prediction serving (1 task)

---

## Wave 0 — Project scaffolding

### [x] T0.1 — Config and environment setup
- **Deliverables:**
  - `config/roster_overrides.example.json` — documented example of the
    manual override schema (§6.2): player in/out flags, team reassignment,
    effective prediction date. Actual `config/roster_overrides.json` stays
    gitignored per existing `.gitignore`.
  - `src/nba_game_projections/config.py` — central config module: data
    directories (`data/raw/`, `data/processed/`), hoopR base URL, season
    range (2017–present), roster-override file path. No hardcoded paths
    elsewhere in the codebase.
  - `pyproject.toml`: add `[tool.ruff]` and `[tool.pytest.ini_options]`
    sections (test paths, lint rules) if not already present.
- **Depends on:** nothing.
- **Blocks:** everything that reads paths/season range (most of Wave 1+).

---

## Wave 1 — Data acquisition and loading

### [x] T1.1 — hoopR downloader
- **File:** `src/nba_game_projections/data/download.py`
- **Scope:** fetch `player_box`, `team_box`, `schedules`, `betting_lines`
  raw files from `raw.githubusercontent.com/sportsdataverse/hoopR-nba-data`
  for seasons 2017–present into `data/raw/`, skipping files already cached.
  Filter `schedules` to four-digit-season filenames only (§4, malformed
  placeholder files noted in PRD). Idempotent/re-runnable.
- **Depends on:** T0.1 (config for paths/season range).
- **Tests:** mock HTTP responses; verify correct URL construction, caching
  skip-if-exists behavior, and malformed-filename filtering.

### [x] T1.2 — Schedules loader
- **File:** `src/nba_game_projections/data/schedules.py`
- **Scope:** load cached `schedules` parquet files into a single validated
  DataFrame (game id, date, home/away team, final score, season).
- **Depends on:** T1.1 (file format/naming contract only — can be stubbed
  with a fixture parquet file and built in parallel with T1.1 finishing).
- **Tests:** load a small fixture parquet, assert schema/dtypes.

### [x] T1.3 — Player box loader
- **File:** `src/nba_game_projections/data/player_box.py`
- **Scope:** load cached `player_box` parquet files into a validated
  DataFrame (per-player per-game stats, `team_id`, minutes, starter flag,
  DNP reason).
- **Depends on:** T1.1 (contract only, same parallelization note as T1.2).
- **Tests:** fixture-based schema/dtype checks.

### [x] T1.4 — Team box loader
- **File:** `src/nba_game_projections/data/team_box.py`
- **Scope:** load cached `team_box` parquet files into a validated
  DataFrame (per-team per-game aggregate stats).
- **Depends on:** T1.1 (contract only).
- **Tests:** fixture-based schema/dtype checks.

### [x] T1.5 — Betting lines loader
- **File:** `src/nba_game_projections/data/betting_lines.py`
- **Scope:** load the single `betting_lines` JSON archive into a validated
  DataFrame keyed by game (closing spread, over/under, favorite/underdog
  cover result), for reporting use only (§8).
- **Depends on:** T1.1 (contract only).
- **Tests:** fixture-based schema checks.

> T1.2–T1.5 can all be built in parallel with each other, and in parallel
> with the tail end of T1.1 once the on-disk file naming/format is agreed.

---

## Wave 2 — Feature engineering

### [x] T2.1 — Roster-aware trailing rolling stats
- **File:** `src/nba_game_projections/features/rolling_stats.py`
- **Scope:** for a given team and as-of date, compute trailing-N-game
  rolling per-player stats (points, rebounds, assists, shooting efficiency,
  minutes share), aggregated to a team-level vector using each player's
  most recent `team_id` (§5) — so trades/injuries flow through automatically
  via the rolling window, no explicit trade-tracking logic.
- **Depends on:** T1.3 (player box loader).
- **Tests:** synthetic multi-team dataset including a mid-window trade;
  assert the traded player's stats move to the new team's aggregate and
  drop out of the old team's within the trailing window.

### [x] T2.2 — Team contextual features
- **File:** `src/nba_game_projections/features/context.py`
- **Scope:** home/away flag, rest days since last game, back-to-back flag,
  and similar schedule-derived features per team per game.
- **Depends on:** T1.2 (schedules loader).
- **Tests:** synthetic schedule with a back-to-back and a bye; assert
  correct rest-day counts and flags.

### [x] T2.3 — Elo rating engine
- **File:** `src/nba_game_projections/features/elo.py`
- **Scope:** game-by-game Elo rating computed across full history
  (K-factor, initial rating, home-court adjustment as tunable parameters
  per §5/§11), exposing per-team pre-game rating as a feature and usable
  standalone as a backtest baseline (§8 baseline #3).
- **Depends on:** T1.2 (schedules loader, for game order and results).
- **Tests:** known small sequence of games with hand-computed expected
  rating updates.

### [x] T2.4 — Roster override loader
- **File:** `src/nba_game_projections/features/roster_overrides.py`
- **Scope:** parse `config/roster_overrides.json` (schema from T0.1) and
  apply in/out or team-reassignment overrides for a specific prediction
  date, producing an adjusted active-player set for T2.1's aggregation
  step to consume instead of the automatic trailing-roster default (§6.2).
- **Depends on:** T0.1 (schema).
- **Tests:** sample override file; assert in/out and reassignment cases
  correctly adjust the active-roster set.

### [x] T2.5 — Feature assembly
- **File:** `src/nba_game_projections/features/build.py`
- **Scope:** combine T2.1 (with optional T2.4 override), T2.2, and T2.3
  into a single per-game, per-team training-ready feature row; produces the
  full feature matrix for a date range.
- **Depends on:** T2.1, T2.2, T2.3, T2.4.
- **Tests:** integration test over a small synthetic multi-season dataset
  asserting the assembled feature matrix shape/columns and no leakage
  (features for a game use only data strictly before that game's date).

> T2.1, T2.2, T2.3, T2.4 can all be built in parallel — each depends only on
> a Wave 1 loader or on T0.1. T2.5 is the integration point and depends on
> all four.

---

## Wave 3 — Model

### [x] T3.1 — Training entrypoint
- **File:** `src/nba_game_projections/models/train.py`
- **Scope:** `train(through_date)` — trains a single gradient-boosted tree
  regressor (LightGBM first; structure the wrapper so XGBoost can be
  swapped in for comparison per §7/§11) on all feature rows strictly before
  `through_date`, predicting margin of victory. Returns a fitted model
  object usable for both backtesting (§8) and future periodic retraining
  (§7).
- **Depends on:** T2.5 (feature assembly).
- **Tests:** train on a small synthetic dataset; assert no rows on/after
  `through_date` were used (leakage check) and that predict() runs.

### [x] T3.2 — Win probability derivation
- **File:** `src/nba_game_projections/models/win_probability.py`
- **Scope:** convert a predicted margin into a win probability using the
  model's historical residual error distribution (normal/logistic link per
  §3).
- **Depends on:** T3.1 (needs a fitted model's residuals, but the
  probability-link function itself can be built/tested against a
  synthetic residual distribution in parallel with T3.1).
- **Tests:** known residual distribution + margin → expected probability
  within tolerance; monotonicity check (larger margin → higher probability).

---

## Wave 4 — Backtest and reporting

### [x] T4.1 — Walk-forward backtest harness
- **File:** `src/nba_game_projections/backtest/walk_forward.py`
- **Scope:** expanding-window backtest driver — repeatedly calls
  `train(through_date)` (T3.1) with season-boundary cutoffs, predicts the
  next season, rolls forward, per §8. Determines the first valid test
  season from available trailing history (§11).
- **Depends on:** T3.1, T2.5.
- **Tests:** small synthetic multi-season dataset; assert correct
  train/test partitioning per fold and that no fold leaks future data.

### [x] T4.2 — Baseline predictors
- **File:** `src/nba_game_projections/backtest/baselines.py`
- **Scope:** four baseline predictors evaluated on the same test folds
  (§8): always-home-team, better-record-entering-game, Elo-alone, and
  Vegas-closing-line-implied-pick.
- **Depends on:** T1.2 (schedules, for record), T2.3 (Elo), T1.5 (betting
  lines).
- **Tests:** each baseline against a small fixture dataset with known
  expected picks.

### [x] T4.3 — Metrics and reporting
- **File:** `src/nba_game_projections/backtest/report.py`
- **Scope:** compute win/loss accuracy per test season and pooled across
  seasons (headline metric, §8), plus secondary margin-error and win-
  probability calibration stats, for both the model and each baseline.
  Produces the final comparison report.
- **Depends on:** T4.1, T4.2, T3.2 (for win-probability calibration stats).
- **Tests:** known prediction/actual arrays → expected accuracy and error
  metrics.

> T4.2 can be built in parallel with T4.1 (both depend only on Wave 1–2
> outputs). T4.3 is the integration point depending on both.

---

## Phase 1 dependency summary

```
T0.1
 ├─> T1.1 ─┬─> T1.2 ─┬─> T2.2 ─┐
 │         ├─> T1.3 ─┼─> T2.1 ─┤
 │         ├─> T1.4  │         ├─> T2.5 ─> T3.1 ─┬─> T3.2 ─┐
 │         └─> T1.5 ─┼─> T2.3 ─┤                 │         │
 │                   │         │                 ├─> T4.1 ─┤
 └─> T2.4 ───────────┴─────────┘                 ├─> T4.2 ─┼─> T4.3
                                                  └─────────┘
```

Widest parallelization points: Wave 1 loaders (T1.2–T1.5), Wave 2 feature
modules (T2.1–T2.4), and Wave 4's T4.1/T4.2.

---

# Phase 2 — Application implementation tasks

Derived from `PRD.md` §12-20. Every design decision referenced below (SQLite
over Postgres, admin-triggered over automated scheduling, current-state-only
overrides, Swagger-only interface, etc.) was made explicitly during scoping -
see the relevant PRD subsection for the "why," not repeated here.

## Wave 5 — App scaffolding, persistence, and independent prep

### [x] T5.1 — FastAPI app + SQLite/SQLAlchemy setup
- **File:** `src/nba_game_projections/app/main.py`, `src/nba_game_projections/app/db.py`
- **Scope:** FastAPI app instance; SQLAlchemy engine pointed at a new
  gitignored SQLite file (e.g. `data/processed/app.db`, add the path to
  `config.py` alongside the other data paths, per PRD §14/§9's config-driven
  preference); session dependency; startup migration/`create_all` so a fresh
  clone is reproducible without a manual DB-setup step. A trivial health-check
  endpoint to prove the app boots.
- **Depends on:** T0.1 (existing `config.py` - extend it, don't fork it).
- **Tests:** app starts via FastAPI's `TestClient`; DB tables exist after
  startup; health-check returns 200.

### [x] T5.2 — Roster override repository
- **File:** `src/nba_game_projections/app/repositories/roster_overrides.py`
- **Scope:** SQLAlchemy model for the current-state roster-override table
  (§15: `player_id`, `player_name`, `team_id`, `status` in/out/reassigned,
  `new_team_id`, `note` - no `prediction_date`, unlike T2.4's file-based
  schema). A repository/factory layer (per standing Factory-pattern
  preference) with list/add/update/remove operations - no HTTP here, just the
  data-access layer T7.1's endpoints will call.
- **Depends on:** T5.1 (DB engine/session).
- **Tests:** CRUD operations against a real (test-scoped) SQLite DB - add,
  update an existing player's status, remove, list reflects current state.

### [x] T5.3 — Model version persistence layer
- **File:** `src/nba_game_projections/app/repositories/model_versions.py`
- **Scope:** SQLAlchemy model for model-version metadata (id, artifact file
  path, `through_date`, `created_at`, `is_active`); save a `TrainedModel`
  (T3.1's dataclass - model, feature columns, residuals, through_date) as a
  versioned `joblib` artifact under `data/processed/models/` without deleting
  prior versions; activate/deactivate so exactly one version is active at a
  time (§14 - "a bad retrain is recoverable by flipping the active pointer
  back, without retraining").
- **Depends on:** T5.1 (DB engine/session), T3.1 (`TrainedModel` shape).
- **Tests:** save two versions, activate the second, assert only it is
  active; load the active version back and assert it round-trips (model
  still predicts, residuals/feature_columns intact).

### [x] T5.4 — Upcoming-games loader
- **File:** `src/nba_game_projections/data/upcoming_games.py`
- **Scope:** load a not-yet-completed season's cached schedule file and
  return its unplayed games (game_id, date, home/away team) using
  `status_type_completed=False` as the signal, since those files lack
  `home_winner`/`away_winner` entirely and carry placeholder `0` scores (§16,
  verified against the real 2026-27 schedule file). Deliberately a **new,
  separate** function from `data/schedules.py`'s `load_schedules()` - that
  loader is not touched, so every existing Wave 1-4 caller's "every row is a
  completed game with a real score" guarantee stays intact.
- **Depends on:** nothing new - reads the same cached parquet files
  `download.py` (T1.1) already fetches.
- **Tests:** fixture-based, a synthetic schedule file shaped like the real
  future-season file (no `home_winner` column, `status_type_completed`
  present) plus a synthetic completed-game file for contrast; assert only
  the unplayed rows come back, with the right columns.

### [x] T5.5 — Make home/away pairing logic public
- **File:** `src/nba_game_projections/models/train.py`
- **Scope:** rename `_wide_game_matrix` to a public `wide_game_matrix` (same
  pattern already used once in Phase 1 - T2.5 made `rolling_stats.py`'s
  roster logic public for the same cross-module-reuse reason) so T8.1's
  slate-prediction code can pair a day's upcoming home/away rows into model
  input without duplicating that logic. Purely mechanical - no behavior
  change.
- **Depends on:** T3.1 (existing function).
- **Tests:** none new - existing Wave 3/4 tests (`test_train.py`,
  `test_walk_forward.py`) must still pass unchanged after the rename,
  proving no behavior changed.

> T5.2 and T5.3 both only depend on T5.1's DB setup and can be built in
> parallel with each other. T5.4 and T5.5 have no Phase-2-specific
> dependencies at all and can be built any time, in parallel with all of
> Wave 5.

---

## Wave 6 — Model lifecycle and admin operations

### [x] T6.1 — Production training entrypoint
- **File:** `src/nba_game_projections/app/services/training.py`
- **Scope:** calls Phase 1's existing `train(through_date)` (T3.1, unchanged)
  with `through_date = now` and `seasons = config.season_range()` (§19 - no
  per-retrain season selection in this pass), then persists the result via
  T5.3's repository and activates it.
- **Depends on:** T5.3 (persistence layer), T3.1 (`train()`).
- **Tests:** mock `train()` to return a stub `TrainedModel` (avoid a real
  ~hour-long fit in tests, per §8.1's measured runtime); assert the result
  gets persisted and activated correctly.

### [x] T6.2 — Data refresh endpoint
- **File:** `src/nba_game_projections/app/routers/admin.py`
- **Scope:** `POST /admin/refresh-data` - calls the existing `download.py`
  (T1.1) functions for the current season(s). **Synchronous** (§16 - cheap,
  idempotent-skip-if-cached, expected to take seconds, unlike retrain).
- **Depends on:** T5.1 (FastAPI app), T1.1 (existing downloader).
- **Tests:** mock the download functions, assert the endpoint calls them and
  returns success; no real network calls in tests.

### [x] T6.3 — Retrain endpoint: background task + status
- **File:** `src/nba_game_projections/app/routers/admin.py`
- **Scope:** `POST /admin/retrain` kicks off T6.1's production training as a
  FastAPI `BackgroundTasks` job and returns a job id immediately (§14 - never
  block a request on a training run); `GET /admin/retrain/{job_id}` reports
  running/done/failed. Retraining does **not** call the data-refresh
  endpoint automatically first (§19 - two independent admin actions).
- **Depends on:** T5.1 (FastAPI app), T6.1 (production training).
- **Tests:** trigger the endpoint with T6.1 mocked, assert it returns
  immediately with a job id and the status endpoint reflects the job's
  state transitions.

### [x] T6.4 — Model admin endpoints
- **File:** `src/nba_game_projections/app/routers/admin.py`
- **Scope:** `GET /admin/models` (list versions, flag the active one),
  `POST /admin/models/{id}/activate` (roll back/switch active model).
- **Depends on:** T5.3 (persistence layer).
- **Tests:** list reflects saved versions; activating a non-active version
  flips the active flag and deactivates the previous one.

> T6.2 depends only on T5.1; T6.4 depends only on T5.3 - both can be built in
> parallel with T6.1/T6.3's training-dependent chain.

---

## Wave 7 — Roster override admin API

### [x] T7.1 — Roster override CRUD endpoints
- **File:** `src/nba_game_projections/app/routers/admin.py`
- **Scope:** `GET /admin/roster-overrides` (list current state),
  `POST /admin/roster-overrides` (add/update one),
  `DELETE /admin/roster-overrides/{id}` (remove one) - thin HTTP layer over
  T5.2's repository.
- **Depends on:** T5.2 (roster override repository).
- **Tests:** FastAPI `TestClient` round-trip - add, list shows it, update
  changes status, delete removes it.

### [x] T7.2 — SQLite-to-RosterOverride adapter
- **File:** `src/nba_game_projections/app/services/roster_overrides.py`
- **Scope:** converts T5.2's current-state SQLite rows into the same
  `list[RosterOverride]` shape T2.4's `apply_roster_overrides()` (Phase 1,
  unchanged) already expects - only the *source* of the override list
  changes (SQLite instead of a `prediction_date`-scoped JSON file), not how
  it's applied to a computed roster.
- **Depends on:** T5.2 (repository), T2.4 (existing `RosterOverride`
  dataclass and `apply_roster_overrides()`).
- **Tests:** given a set of SQLite rows (in/out/reassigned mix), assert the
  produced `list[RosterOverride]` matches what T2.4's own tests already
  assert `apply_roster_overrides()` correctly handles.

> T7.1 and T7.2 both depend only on T5.2 and can be built in parallel.

---

## Wave 8 — Prediction serving

### [x] T8.1 — Slate prediction endpoint
- **File:** `src/nba_game_projections/app/routers/predictions.py`,
  `src/nba_game_projections/app/services/prediction.py`
- **Scope:** `GET /predictions/slate?date=YYYY-MM-DD` (defaults to today,
  §17/§19 - the sole query shape for this pass). For the given date: load
  that day's games (T5.4's upcoming-games loader), build features via the
  existing `build_feature_matrix` (T2.5, unchanged) using T7.2's live
  overrides instead of the file-based default, pair into model-input rows
  via T5.5's now-public helper, run inference through the currently-active
  model (T5.3), and derive win probability via T3.2's `win_probability`
  (unchanged). Computed **fresh on every request** - no caching (§17), so a
  roster-override edit changes the very next request's probabilities with no
  separate recompute step.
- **Depends on:** T5.3 (active model), T5.4 (upcoming games), T5.5 (pairing
  helper), T7.2 (live overrides), T3.2 (`win_probability`).
- **Tests:** integration test with the active model, upcoming-games loader,
  and override repository all mocked/fixture-backed - assert the response
  shape (predicted margin, predicted winner, win probability per game) and,
  critically, that changing a fixture override between two calls changes the
  second call's win probability (proves the no-caching design actually
  works end to end, not just that it was intended to).

> This is Phase 2's integration point, analogous to Phase 1's T2.5/T4.3 -
> it depends on nearly everything built in Waves 5-7.

---

## Phase 2 dependency summary

```
T5.1 ─┬─> T5.2 ─┬─> T7.1
      │         └─> T7.2 ───────┐
      ├─> T5.3 ─┬─> T6.1 ─> T6.3 │
      │         └─> T6.4         │
      └─> T6.2                   │
                                  ├─> T8.1
T5.4 ─────────────────────────────┤
T5.5 ─────────────────────────────┘
```

Widest parallelization points: T5.2/T5.3/T5.4/T5.5 within Wave 5 (all
independent of each other beyond T5.1), and T6.2/T6.4/T7.1/T7.2 across Waves
6-7 (each depends on only one Wave 5 output). T8.1 is the single integration
point everything else feeds into, same role T2.5/T4.3 played in Phase 1.
