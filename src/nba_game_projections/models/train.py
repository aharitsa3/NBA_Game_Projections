"""Training entrypoint (T3.1): `train(through_date)` fits a margin-of-victory regressor.

`build_feature_matrix` (T2.5) produces one row per *team* per game — each
row carries only that team's own features, not its opponent's. Predicting
a game's margin needs both sides at once, so this module's real job is
widening that long, per-team matrix into one row per *game* (`home_*` and
`away_*` columns side by side) before fitting.

The regressor itself (LightGBM) is isolated in `_fit_model` so it can be
swapped for XGBoost later (PRD §7/§11) without touching the assembly or
leakage-guard logic above it.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from nba_game_projections import config
from nba_game_projections.data.schedules import load_schedules
from nba_game_projections.features.build import build_feature_matrix

_BASE_FEATURES = (
    "rest_days",
    "is_back_to_back",
    "elo_rating",
    "rolling_points",
    "rolling_rebounds",
    "rolling_assists",
    "rolling_steals",
    "rolling_blocks",
    "rolling_turnovers",
    "rolling_efg_pct",
    "rolling_minutes_per_game",
    "rolling_roster_size",
)

FEATURE_COLUMNS = tuple(f"home_{c}" for c in _BASE_FEATURES) + tuple(
    f"away_{c}" for c in _BASE_FEATURES
)

_DEFAULT_MODEL_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "verbosity": -1,
}


@dataclass
class TrainedModel:
    model: lgb.LGBMRegressor
    feature_columns: tuple[str, ...]
    residuals: np.ndarray
    through_date: pd.Timestamp

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[list(self.feature_columns)])


def wide_game_matrix(long: pd.DataFrame) -> pd.DataFrame:
    """Pivot T2.5's per-team-per-game rows into one row per game (home_*/away_* columns)."""
    keys = ["game_id", "date", "season"]
    home = long[long["is_home"]]
    away = long[~long["is_home"]]

    home_side = home[[*keys, "team_id", *_BASE_FEATURES]].rename(
        columns={"team_id": "home_team_id", **{c: f"home_{c}" for c in _BASE_FEATURES}}
    )
    away_side = away[["game_id", "team_id", *_BASE_FEATURES]].rename(
        columns={"team_id": "away_team_id", **{c: f"away_{c}" for c in _BASE_FEATURES}}
    )
    return home_side.merge(away_side, on="game_id")


def assemble_training_frame(
    through_date,
    seasons=None,
    rolling_window: int = 10,
    elo_k_factor: float = 20.0,
    elo_initial_rating: float = 1500.0,
    elo_home_advantage: float = 100.0,
) -> pd.DataFrame:
    """Wide, per-game training frame (features + `margin` target) for games before `through_date`.

    History starts at `config.SEASON_START` (the Phase 1 window's earliest
    season) so rolling stats/Elo have as much trailing context as the
    project's data actually supports.
    """
    through_date = pd.Timestamp(through_date)
    start_date = pd.Timestamp(f"{config.SEASON_START}-01-01")

    long = build_feature_matrix(
        start_date,
        through_date,
        seasons=seasons,
        rolling_window=rolling_window,
        elo_k_factor=elo_k_factor,
        elo_initial_rating=elo_initial_rating,
        elo_home_advantage=elo_home_advantage,
    )
    wide = wide_game_matrix(long)

    schedules = load_schedules(seasons)
    scores = schedules[["game_id", "home_score", "away_score"]]
    wide = wide.merge(scores, on="game_id")
    wide["margin"] = wide["home_score"] - wide["away_score"]

    assert (wide["date"] < through_date).all(), (
        "leakage: a training row is not strictly before through_date"
    )
    return wide


def _fit_model(
    X: pd.DataFrame, y: pd.Series, model_params: dict | None = None
) -> lgb.LGBMRegressor:
    params = {**_DEFAULT_MODEL_PARAMS, **(model_params or {})}
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y)
    return model


def train(
    through_date,
    seasons=None,
    rolling_window: int = 10,
    elo_k_factor: float = 20.0,
    elo_initial_rating: float = 1500.0,
    elo_home_advantage: float = 100.0,
    model_params: dict | None = None,
) -> TrainedModel:
    """Train a margin-of-victory regressor on all feature rows strictly before `through_date`.

    Reused unchanged both for the walk-forward backtest (T4.1, called
    repeatedly with increasing cutoffs) and for future periodic retraining
    (PRD §7).
    """
    through_date = pd.Timestamp(through_date)
    frame = assemble_training_frame(
        through_date,
        seasons=seasons,
        rolling_window=rolling_window,
        elo_k_factor=elo_k_factor,
        elo_initial_rating=elo_initial_rating,
        elo_home_advantage=elo_home_advantage,
    )

    X = frame[list(FEATURE_COLUMNS)]
    y = frame["margin"]
    model = _fit_model(X, y, model_params)
    residuals = (y - model.predict(X)).to_numpy()

    return TrainedModel(
        model=model, feature_columns=FEATURE_COLUMNS, residuals=residuals, through_date=through_date
    )
