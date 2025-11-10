from typing import Protocol, Dict, Any
from booster.booster import GradientBoosterRegressor
import numpy as np

class Callback(Protocol):
    def on_iteration_end(self, booster: Any,
                         iter_idx: int, evals: Dict[str, float]) -> None: ...

class EarlyStopping:
    def __init__(self, X_val, y_val, patience: int, min_delta, mode='min'):
        assert mode in ('min', 'max'), "mode should be 'min' or 'max'"
        self.X_val, self.y_val = X_val, y_val
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = np.inf if mode == 'min' else -np.inf
        self.best_iter = -1
        self.wait = 0
        self.metric_name = 'val_loss'

    def _is_better(self, score: float) -> bool:
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta
        
    def on_iteration_end(self, booster: GradientBoosterRegressor,
                         iter_idx: int, evals: Dict[str, float]) -> None:
        y_pred = booster.predict(self.X_val)
        loss = booster.objective.loss(self.y_val, y_pred)
        evals[self.metric_name] = loss

        if self._is_better(loss):
            self.best_score = loss
            self.best_iter = iter_idx
            self.wait = 0
            evals['best_score'] = self.best_score
            evals['best_iter'] = self.best_iter
            evals['wait'] = self.wait
        else:
            self.wait += 1
            evals['wait'] = self.wait
            if self.wait >= self.patience:
                booster.stop_training = True
                print(f"Early stopping at iteration {iter_idx}. Best {self.metric_name}: {self.best_score} at iteration {self.best_iter}")
        
        return evals