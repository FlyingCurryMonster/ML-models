"""
GGBM: Generalized Gradient Boosting for Any Model

A flexible gradient boosting framework that can boost any model architecture.
"""

__version__ = "0.1.0"

from .booster import GeneralizedGradBoostingRegressor
from .objectives import (
    Objective,
    SquaredError,
    AbsoluteError,
    HuberLoss,
    QuantileLoss,
)
from .callbacks import Callback, EarlyStopping

# Convenient aliases
GGBM = GeneralizedGradBoostingRegressor
GGBMRegressor = GeneralizedGradBoostingRegressor

__all__ = [
    "GeneralizedGradBoostingRegressor",
    "GGBM",
    "GGBMRegressor",
    "Objective",
    "SquaredError",
    "AbsoluteError",
    "HuberLoss",
    "QuantileLoss",
    "Callback",
    "EarlyStopping",
]
