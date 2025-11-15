"""
Generalized Additive Models (GAM)

Simple and efficient GAM implementations using sklearn components.
Designed to work seamlessly with GGBM for boosting.
"""

from .sklearn_gam import SklearnGAM

# Also export statsmodels version (experimental)
try:
    from .statsmodels_gam import StatsmodelsGAM
    __all__ = ["SklearnGAM", "StatsmodelsGAM"]
except ImportError:
    __all__ = ["SklearnGAM"]
