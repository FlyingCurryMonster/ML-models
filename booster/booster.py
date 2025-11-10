# booster.py
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from objectives import Objective
from callbacks import Callback

class GradientBoosterRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, base_learner, n_estimators, learning_rate=0.1,
                 objective: Objective=None, second_order=True, subsample=1.0,
                 random_state=None, callbacks: list[Callback]=None):
        
        assert subsample <=1.0 and subsample > 0.0, "subsample must be in (0.0, 1.0]"

        self.base_learner = base_learner
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.objective = objective
        self.second_order = second_order
        self.subsample = subsample
        self.random_state = random_state
        self.callbacks = callbacks if callbacks is not None else [Callback]
        self._learners = []
    
    def fit(self, X, y, sample_weight=None, X_val=None, y_val=None):
        if self._learners is not None:
            self._learners = []
        self.stop_training = False

        rng = np.random.default_rng(self.random_state)
        obj = self.objective
        y_pred = obj.init_prediction(y, sample_weight=sample_weight)
        self.init_ = y_pred
        evals = []


        for t in range(self.n_estimators):
            
            # pseudo-residuals
            grad, hess = obj.gradient(y, y_pred)
            
            if not self.second_order:
                pseudo_resid = -grad
                w = None

            else:
                pseudo_resid = -grad / np.clip(hess, a_min=1e-12, a_max=None)
                if sample_weight is None:
                    w = hess
                else:
                    w = sample_weight * hess

            # Subsampling
            if self.subsample < 1.0:
                n = len(y)

                # draw training examples without replacement
                idx = rng.choice(n, 
                                 # ensure at least one sample, and subsample not more than n
                                 min(n, max(int(self.subsample * n), 1)),
                                 replace=False)
                X_t, pseudo_rt = X[idx], pseudo_resid[idx]
                w_t = None if w is None else w[idx]
            else:
                X_t, pseudo_rt, w_t = X, pseudo_resid, w

            # fit the learner
            learner = self._clone_base()
            learner.fit(X_t, pseudo_rt, sample_weight=w_t)
            self._learners.append(learner)

            # update predictions
            y_pred += self.learning_rate * learner.predict(X)

            # callbacks (logging/early stopping)
            eval = {}
            if self.callbacks is not None:
                for cb in self.callbacks:
                    eval = cb.on_iteration_end(self, t, eval)
            evals.append(eval)

            if self.stop_training:
                break

        return self


    def predict(self, X):
        base = getattr(self, "init_", 0.0)
        out = np.full(shape=(len(X),), fill_value=base, dtype=np.float64)
        
        for f in getattr(self, "_learners", []):
            out += self.learning_rate * f.predict(X)
        
        return out

    def _clone_base(self):
        # shallow clone; for sklearn estimators use sklearn.clone
        import copy
        return copy.deepcopy(self.base_learner)