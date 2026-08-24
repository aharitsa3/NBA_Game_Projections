# NBA Game Projections — Product Requirements Document

Status: Phase 1 complete (built and validated via full 2017-2026 walk-forward
backtest — see §8 Results). Phase 2 (application) scoped in detail below
(§12-20), not yet built.
Last updated: 2026-08-20

## 1. Purpose

Build a machine-learning system that predicts NBA game outcomes, using both
team-level and individual-player-level historical performance, sensitive to
in-season roster changes (trades, injuries). This is a research project, not
a commercial product — the goal is to prove the approach can predict games
competitively, not to monetize it.

The project is split into two phases:

- **Phase 1 (complete):** prove feasibility with a pure ML/data pipeline —
  no application layer. Success is judged by walk-forward backtest results,
  not a fixed target set in advance. See §8 for the actual results.
- **Phase 2 (scoped, not yet built):** package the proven model into an
  application (FastAPI backend, no frontend in this first pass — see
  §12-20) for generating and viewing predictions, with admin-editable
  roster overrides feeding back into win probability.

This document covers Phase 1 in full detail (§1-11) and Phase 2 in full
detail (§12-20), the latter scoped only after Phase 1 results existed, per
the original plan above.

## 2. Non-goals (explicitly out of scope for Phase 1)

- No web app, API, or UI of any kind.
- No betting/monetization functionality.
- No live/automated injury-report ingestion (see §6.3 — manual override only).
- No two-stage per-player production model or full RAPM/lineup-based player
  rating system (may be revisited in a later phase; see §10).
- No automated "beat the market" optimization against Vegas lines — Vegas
  data is used only as a reporting baseline (§8), not a training target.

## 3. Prediction target

- **Core model output:** predicted margin of victory (home team score minus
  away team score), via regression.
- **Derived output:** win probability, computed from the predicted margin
  and the model's residual error distribution (e.g. via a normal/logistic
  link over the margin's predicted value and historical residual variance).
- **Headline reported metric:** win/loss accuracy (does the sign of the
  predicted margin match the actual winner). Chosen deliberately over
  MAE-vs-Vegas as the headline number — the user will do detailed
  sportsbook-line comparisons manually, outside this pipeline.

## 4. Data source

**Primary and only data source: [hoopR-nba-data](https://github.com/sportsdataverse/hoopR-nba-data)**
(sportsdataverse), a community-maintained, actively-updated archive of NBA
game data derived from ESPN, licensed CC BY 4.0. Verified during scoping
(2026-08-18):

| Folder | Contents | Granularity | Notes |
|---|---|---|---|
| `player_box` | Per-player, per-game box scores (57 columns: points, rebounds, assists, shooting splits, minutes, starter flag, position, `team_id`, DNP reason, etc.) | One parquet file per season, 2002–present | **Primary source of roster composition** — a player's `team_id` on a given game date reflects trades automatically. |
| `team_box` | Per-team, per-game aggregate stats (pace-adjacent: fast break points, points in paint, rebounding, turnovers, shooting splits) | One parquet file per season, 2002–present | |
| `schedules` | Game schedule/results | One parquet file per season (`nba_schedule_{season}.parquet`), 2002–present | Some malformed placeholder files exist in the repo (e.g. two-digit-named files) — filter to four-digit season names only. |
| `betting_lines` | Historical closing spread, over/under, favorite/underdog cover result | Single JSON archive, all seasons from 2017 | Used only for the Vegas-implied-pick reporting baseline (§8), never as a training feature or target. |
| `rosters` | Current-roster snapshot | Only 2025/2026 — **not season-partitioned history** | **Not usable for historical training** (verified 2026-08-18). Roster composition instead derived from `player_box`. |

Data is fetched via direct HTTPS download of the raw parquet/JSON files
(`raw.githubusercontent.com`), cached locally under `data/raw/` (gitignored,
reproducible by re-running the downloader), no scraping or paid API needed.

### 4.1 Historical window

**Seasons 2017–present** for the Phase 1 POC. Rationale: betting_lines data
(the reporting baseline) only goes back to 2017, so starting there keeps
every training/test season comparable against a market baseline. Extending
back to 2004 (hoopR data supports it) is a documented future option if more
training history is needed later (§10) — pre-2017 seasons would simply lack
a Vegas-implied baseline row.

## 5. Feature architecture

**Approach: team-level rolling aggregates built from roster-aware player
box scores** (chosen over a two-stage per-player production model or a full
RAPM/lineup-rating system, for POC speed — see §10 for the deferred
alternatives and why).

For each upcoming game, each team's feature vector is built by aggregating
its *currently active* players' recent per-game/per-minute production:

- Trailing-N-game rolling stats per player (points, rebounds, assists,
  shooting efficiency, etc.), weighted by recent minutes share, summed/
  averaged up to a team-level vector.
- Because aggregation reads each player's current `team_id` from the box
  score data, a mid-season trade is reflected automatically — the traded
  player's production drops out of the old team's aggregate and into the
  new team's within the trailing window, with no separate trade-tracking
  logic required.
- Team-level contextual features: home/away, rest days, back-to-back flag,
  and similar schedule-derived signals.
- **Elo rating**: a simple Elo-style team rating, updated game-by-game
  across the full history, included as one input feature to the model (not
  a separate model stage — see §6).

Exact rolling-window sizes and Elo parameters (K-factor, initial ratings,
home-court adjustment) are implementation details to be tuned during
building, not fixed in advance by this document.

## 6. Roster / player-availability handling

Two layers, per the agreed design:

1. **Automatic default — trailing-active-roster.** A player is treated as
   part of a team's current rotation if they've appeared in that team's
   recent games (within the trailing window used for feature aggregation).
   This requires no extra data source and handles trades and multi-week
   injuries/absences reasonably well as a byproduct of the rolling window.
2. **Manual override.** A configuration (`config/roster_overrides.json`,
   gitignored — personal/ephemeral prediction input, not project data) lets
   the user mark specific players in/out, or reassign a player to a
   different team, for a specific prediction date. Editing this file and
   re-running the prediction for that date recomputes the affected team's
   feature vector using the override instead of the automatic default.

**Explicitly out of scope for Phase 1:** a live/automated injury-report
feed (e.g. NBA official injury report, ESPN, Rotowire). This is a
documented Phase 2+ enhancement (§10) — same-day injury news is a
refinement on a working model, not a prerequisite for proving feasibility.

## 7. Model

- **Algorithm:** a single gradient-boosted tree regressor (LightGBM or
  XGBoost — final choice made during implementation based on which
  performs/tunes better on this data) predicting margin of victory.
- **Elo as a feature, not a separate stage:** rather than a two-stage
  Elo-then-GBT hybrid, Elo rating is computed once per team per game and
  fed into the GBT as one input feature alongside the roster-aggregate and
  contextual features. This keeps the pipeline to a single model while
  still giving the GBT a strong, well-understood prior signal to build on.
- **Training entrypoint is cutoff-date-parameterized from the start:** a
  single function, `train(through_date)`, trains on all data strictly
  before `through_date`. This function serves double duty:
  - Called repeatedly with increasing cutoffs to run the walk-forward
    backtest (§8).
  - Reused unchanged for future periodic retraining once real predictions
    are being generated (see §6 and §9 — retrain cadence is periodic, e.g.
    weekly/monthly in-season plus each new season, while per-game features
    like rolling stats and Elo are recomputed continuously without needing
    a model retrain).

## 8. Validation and reporting

- **Protocol: strict walk-forward (expanding-window) backtesting.** Train
  on all seasons strictly before a cutoff, test on the next season, roll
  the cutoff forward one season at a time, repeat. No random/k-fold
  splits — those would leak future information into training via
  rolling-window features and produce misleadingly optimistic results.
  First test fold will necessarily be a season or two into the 2017+
  window, once enough trailing history exists to compute meaningful
  rolling features.
- **Headline metric:** win/loss accuracy per test season (and pooled
  across all test seasons).
- **Secondary outputs:** predicted margin (with error distribution) and
  derived win probabilities, retained for future calibration analysis even
  though they're not the headline metric.
- **Reported baselines**, computed on the same walk-forward test folds for
  direct comparison:
  1. Always pick the home team.
  2. Always pick the team with the better record entering the game.
  3. Elo rating alone, as a standalone predictor (isolates how much the
     richer feature set adds over a simple rating system).
  4. Vegas closing-line implied pick, from `betting_lines` (reference only
     — not a target the pipeline optimizes against).
- **No fixed numeric success threshold set in advance.** Results will be
  reviewed jointly against the above baselines once the backtest runs,
  rather than judged against a pre-committed target accuracy.

### 8.1 Results (full 2017-2026 backtest, run 2026-08-19)

11,627 test games across 9 seasons (2018-2026; 2017 used only as the first
training season, per §11's resolution of the first-valid-test-season
question). Pooled win/loss accuracy:

| Predictor | Pooled accuracy |
|---|---|
| Vegas favorite | 66.8% (2018-2023 only - see below) |
| **Model** | **65.6%** |
| Elo alone | 63.6% |
| Better record | 63.0% |
| Always home | 55.8% |

The model beats every baseline except Vegas, consistently across all 9
seasons individually, not just pooled. It doesn't beat Vegas, which is
expected and fine (§2, §8 - Vegas is a reference baseline, never an
optimization target).

**Caveat on the Vegas row:** the `betting_lines` archive only covers
through the 2022-23 season (last game date 2023-06-12) - hoopR's 2023-24
season onward has no betting-lines data yet. So the 66.8% figure is really
over 2018-2023 only, not the full pooled window; the model-vs-Vegas
comparison over that same 2018-2023 subset is closer (65.4% vs 66.8%).
This is a real gap in the upstream archive, not a join bug (confirmed by
inspecting the loaded data directly).

**Calibration:** Brier score 0.232. The predicted-probability-vs-realized
-win-rate table tracks well in the middle (0.4-0.7 predicted buckets), but
the model is overconfident at both tails - e.g. games it rated ~96% likely
only won ~79% of the time, and games rated ~4% likely actually won ~29% of
the time. Not surprising for a first pass with zero hyperparameter tuning;
a candidate future refinement (Platt scaling/isotonic regression on top of
the current normal-link) rather than a blocker for Phase 2.

**Known performance limitation carried into Phase 2:** the rolling-stats
feature computation (§5, `features/rolling_stats.py`) rescans the full
`player_box` table per team-game row, undocumented cost at small scale but
significant at full-history scale - the full 9-fold backtest took roughly
an hour. Phase 2's production training (§14) calls the same `train()`
function once per retrain (not 9x), so a single retrain is proportionally
cheaper than the full backtest, but still slow enough that it must run as
a background task (§14), never inline in a request.

## 9. Technical setup

- **Language/runtime:** Python 3.13.
- **Dependency management:** `uv`, `pyproject.toml`-based, chosen for fast,
  easy dependency upgrades.
- **Project structure:** a proper installable package
  (`src/nba_game_projections/`), not notebook-driven — `data/`,
  `features/`, `models/`, `backtest/` submodules. No exploratory notebooks
  as part of the durable pipeline; all logic lives in importable,
  testable, re-runnable code from the start, since Phase 2 will reuse this
  code directly rather than porting it out of notebooks.
- **Core dependencies:** `pandas`, `pyarrow`, `numpy`, `lightgbm`,
  `scikit-learn`, `requests`, `tqdm` (runtime); `pytest`, `ruff` (dev).
- **Data caching:** downloaded hoopR files cached under `data/raw/`
  (gitignored, reproducible via the downloader — not committed to git).

## 10. Deferred / future work (not in Phase 1 scope)

Documented here so they aren't lost, but explicitly not being built now:

- Extending the historical window back to 2004 (more training data, no
  Vegas baseline for pre-2017 seasons).
- Two-stage per-player production model (predict each player's expected
  box score, then aggregate) instead of direct team-level aggregation.
- Full player-rating system (RAPM/real-plus-minus style) built from
  play-by-play/lineup data (`pbp`, `shots` folders already available in
  hoopR-nba-data) — would isolate individual player impact independent of
  teammates, more rigorous than box-score aggregation.
- Live/automated injury-report ingestion, replacing the manual override
  config.
- Automated MAE-vs-Vegas and against-the-spread (ATS) benchmarking (the
  user will do this comparison manually for now).
- Phase 2 application: FastAPI backend (Pydantic models, Factory pattern,
  optional auth middleware, config-driven environment settings per
  standing preference) serving predictions from the trained model,
  optionally an Angular frontend. Architecture to be scoped separately
  once Phase 1 proves the model out.

## 11. Open items to resolve during implementation (not requiring sign-off)

- Exact rolling-window size(s) for player/team aggregates.
- Elo parameters (K-factor, home-court adjustment, initial ratings).
- Final choice between LightGBM and XGBoost.
- Definition of the first valid walk-forward test season (depends on how
  much trailing history the rolling features need).

---

# Phase 2 — Application

Scoped 2026-08-20, via structured interview, after Phase 1 results existed
(§8.1). Everything below is a decision, not a suggestion - each was chosen
explicitly over a stated alternative. Not yet built.

## 12. Purpose and scope

Serve predictions for real, not-yet-played games from a persisted,
periodically-retrained version of the Phase 1 model, with an
admin-editable roster-override system so injuries/trades/other player
movement the model can't yet see (same-day news, not reflected in trailing
box-score data) can be corrected by hand and immediately change the
affected games' predicted win probability.

**Confirmed feasible without a new data source:** hoopR's `schedules`
archive already publishes the full upcoming season's schedule in advance
(verified 2026-08-20 - the 2026-27 season's 1,206 games, Oct 2026-Apr
2027, were already present with real matchups/dates in the archive months
before the season starts). Those rows use `status_type_completed=False`
and placeholder `0` scores instead of the `home_winner`/real-score columns
the Phase 1 `schedules` loader depends on - see §16.

## 13. Non-goals (this pass)

Explicitly deferred, not being built now (see also §20 for the fuller
future list):

- No real authentication/multi-user support - single user, no login
  system. "Admin" means a section of the app, not a login role.
- No frontend - FastAPI's auto-generated Swagger UI (`/docs`) is the only
  interface. A real UI (Angular, per standing preference, or something
  lighter) is explicitly deferred to a later pass, to be configured
  separately.
- No mobile app, native or cross-platform - a web app is fully usable from
  a phone's browser already; there's nothing a native app would add for a
  single-user tool.
- No remote hosting/deployment - runs locally via `uvicorn` only. No
  environment-specific config, secrets management, or HTTPS concerns in
  this pass.
- No automated scheduling infrastructure - retraining and data refresh are
  both admin-triggered actions, never a background cron/scheduled job.
- No override history/timeline - roster overrides are current-state only
  (§15), not date-scoped or retroactively queryable.

## 14. Architecture and model lifecycle

**Stack:** FastAPI backend; SQLite (via SQLAlchemy) for all app-managed
state (roster overrides, model version metadata) - the actual `.db` file
is gitignored (mutable runtime state, same treatment as `data/raw/` and
`config/roster_overrides.json` in Phase 1), while the schema
(SQLAlchemy models/migrations) is committed as code, so a fresh clone is
reproducible via a startup migration. No Docker/Postgres - a single-user
app with a small admin-state table doesn't need a server process, and
SQLAlchemy leaves room to move to Postgres later without a rewrite if that
ever changes.

**Production training** is a thin new entrypoint that calls Phase 1's
existing `train(through_date)` (`models/train.py`, unchanged) with
`through_date = now` and the full `config.season_range()` - not tied to
any backtest fold. This is new *usage* of `train()`, not new training
logic; §3.1/T3.1's function was already designed to serve exactly this
purpose ("reused unchanged for future periodic retraining").

**Model persistence:** each production train writes a new versioned
artifact via `joblib` (serializing the whole `TrainedModel` dataclass -
the LightGBM model, feature columns, residuals, and through_date together
as one object) to `data/processed/models/`, without deleting prior
versions. A SQLite table tracks which version is currently "active." A bad
retrain is recoverable by flipping the active pointer back, without
retraining.

**Retrain trigger and execution:** admin-triggered on demand (no automated
cadence, despite §7's "periodic" framing - you decide when, e.g. weekly).
Given §8.1's confirmed multi-fold backtest runtime (~1 hour) and that a
single production train is proportionally cheaper but still slow, the
retrain endpoint runs as a FastAPI background task and returns
immediately; a separate status endpoint reports running/done/failed.
Retraining and data refresh (§16) are independent admin actions - retrain
does not automatically refresh data first.

**In-memory model caching:** the active model is loaded from disk once
(at app startup, and again whenever a retrain activates a new version),
not reloaded from disk on every prediction request.

## 15. Roster override system

Replaces Phase 1's `config/roster_overrides.json` (§6.2, T2.4) - that file
was a one-shot snapshot, hand-edited and scoped to one `prediction_date`
per edit. This doesn't fit an admin feature used continuously as real
injury/trade news comes in.

**Data model: current state, no history.** A SQLite table holds "what's
true right now" - the same fields as Phase 1's schema, minus
`prediction_date`:

| Field | Notes |
|---|---|
| `player_id` | |
| `player_name` | |
| `team_id` | |
| `status` | `in` / `out` / `reassigned` (same three values as T2.4) |
| `new_team_id` | only for `status=reassigned` |
| `note` | free text |

Every prediction, for any date, reads whatever this table currently says
- there's no "what would this have predicted on a past date" capability,
and an override stays in effect until an admin manually changes or
removes it (e.g. a player marked `out` for an injury needs to be manually
cleared once they return - it isn't time-boxed).

Admin CRUDs this table via API (§18): list current overrides, add/update
one, remove one. T2.4's existing `apply_roster_overrides()` delta logic
(`features/roster_overrides.py`) is reused as-is against whatever the
current SQLite state is at prediction time - only where the override list
comes from changes, not how it's applied to a computed roster.

## 16. Data freshness and the upcoming-games loader

Two distinct freshness needs, addressed differently (§7 already draws
this distinction for the model vs. its features):

**Underlying hoopR data** (schedules/player_box) needs to be reasonably
current for rolling stats on real predictions to mean anything. Refreshing
it is cheap - `data/download.py`'s existing functions already skip
already-cached files, so a refresh is a handful of small HTTP calls for
the current season, expected to take seconds. Admin-triggered,
**synchronous** (unlike retrain - fast enough not to need background-task
treatment).

**Unplayed-game schedule rows need a new, separate loader.** Verified
2026-08-20: a not-yet-played season's schedule file (e.g.
`nba_schedule_2027.parquet`) has a different shape than a completed
season's - no `home_winner`/`away_winner` columns at all (not just null),
and `home_score`/`away_score` are placeholder `0`s for every row;
`status_type_completed=False` is the reliable "not yet played" signal
instead. The existing `data/schedules.py` / `load_schedules()` is **not**
modified to handle this - every Wave 1-4 caller (training, backtesting,
baselines) depends on its current guarantee that every row is a completed
game with a real score, and changing that contract risks a placeholder
score silently leaking into training as a fake 0-0 result. Instead, a new,
separate function (e.g. `load_upcoming_games()`) reads not-yet-completed
rows specifically for the prediction-serving path.

## 17. Prediction serving

`GET /predictions/slate?date=...` (defaults to today) - given a date,
return every game scheduled that day plus each one's prediction (predicted
margin, predicted winner, win probability). This is the app's primary
entrypoint, not a single-game or date-range lookup (both considered and
rejected - see §19 for why range queries were deferred, not ruled out).

Computed **fresh on every request** - no caching, no invalidation logic.
Reads the live roster-override table (§15) and the currently-active model
(§14) at request time, so editing an override changes the next slate
request's win probabilities immediately, with no separate "recompute"
step. A single day's slate (typically 5-15 games) is a small computation -
the ~1 hour cost observed in §8.1 comes from training/backtesting across
years of data, not from inference plus feature computation for one day's
games.

Internally reuses the same `_wide_game_matrix`-style home/away pairing
logic already built for training (`models/train.py`) - that helper will
need to become non-private for this cross-module reuse, the same pattern
already used once in Phase 1 (T2.5 made `rolling_stats.py`'s roster logic
public for the same reason).

## 18. API surface (proposed)

Not final - a starting sketch to build from, not a contract:

| Endpoint | Purpose |
|---|---|
| `GET /predictions/slate?date=YYYY-MM-DD` | §17 - the core prediction view |
| `GET /admin/roster-overrides` | list current overrides |
| `POST /admin/roster-overrides` | add/update one |
| `DELETE /admin/roster-overrides/{id}` | remove one |
| `POST /admin/retrain` | kick off a background production retrain, returns a job id |
| `GET /admin/retrain/{job_id}` | poll retrain job status |
| `POST /admin/refresh-data` | synchronous data refresh (§16) |
| `GET /admin/models` | list model versions and which is active |
| `POST /admin/models/{id}/activate` | roll back/switch the active model |

## 19. Assumptions and defaults (confirmed during scoping, low-stakes)

- Production retrain always uses `config.season_range()` (2017-present),
  matching the backtest's default - no per-retrain season selection in
  this pass.
- `GET /predictions/slate` is the only prediction query shape for now;
  single-game and date-range variants were discussed and deferred (not
  ruled out) since the slate view covers the primary use case.

## 20. Deferred / future work (Phase 2+, beyond this pass)

In addition to §13's non-goals and §10's Phase-1-era deferred list:

- Real authentication/multi-user roles, if this is ever used by more than
  one person.
- A real frontend (Angular, per standing preference, or a lighter
  alternative) once the Swagger-UI-only interface feels limiting.
- Remote hosting/deployment, with accompanying environment/secrets
  config.
- Roster-override history/timeline (retroactive "what would this have
  predicted on date D" queries), if current-state-only turns out to be
  limiting in practice.
- Automated retrain/data-refresh scheduling, if admin-triggered starts to
  feel like a chore.
- Addressing §8.1's rolling-stats performance limitation directly (the
  per-row full-table rescan), if production retrain latency becomes
  actually painful rather than just slow.
