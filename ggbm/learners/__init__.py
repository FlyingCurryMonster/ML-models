"""
Learner adapters for various ML frameworks.
"""

from .base import WeakLearner
from .sklearn_adapter import SklearnRegressorAdapter

__all__ = [
    "WeakLearner",
    "SklearnRegressorAdapter",
]
