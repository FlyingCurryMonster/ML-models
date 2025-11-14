# learners/base.py
from typing import Protocol, Optional
import numpy as np

class WeakLearner(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weight: Optional[np.ndarray]=None) -> "WeakLearner": ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...