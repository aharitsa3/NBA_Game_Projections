"""Win probability derivation from predicted margin (PRD §3).

Uses a normal link: `actual_margin = predicted_margin + residual`, with
`residual ~ Normal(0, sigma)` where `sigma` is the population standard
deviation of the model's historical residuals (actual minus predicted
margin). Population std (`ddof=0`) is used deliberately rather than the
sample correction (`ddof=1`) — training residual arrays have thousands of
games in real use, so the correction is negligible, and population std
avoids `ddof=1`'s NaN-on-single-element edge case. `scipy` is deliberately
avoided (not a declared project dependency) in favor of the standard normal
CDF via `math.erf` from the stdlib.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def normal_cdf(z: float) -> float:
    """Standard normal CDF via math.erf (no scipy dependency)."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def win_probability(predicted_margin: float, residuals: Sequence[float]) -> float:
    """P(home team wins) given a predicted margin and the model's historical residuals."""
    sigma = float(np.std(np.asarray(residuals, dtype=float)))

    if sigma == 0:
        if predicted_margin > 0:
            return 1.0
        if predicted_margin < 0:
            return 0.0
        return 0.5

    return normal_cdf(predicted_margin / sigma)
