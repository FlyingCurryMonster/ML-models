"""
Generalized Gradient Boosting Regressor

A flexible gradient boosting framework that can boost any base learner.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from .objectives import Objective
from .callbacks import Callback


class GeneralizedGradBoostingRegressor(BaseEstimator, RegressorMixin):
    """
    Generalized Gradient Boosting Regressor.

    Implements gradient boosting that works with any base learner,
    supporting both first-order and second-order (Newton) boosting.

    Parameters
    ----------
    base_learner : estimator
        The base model to boost. Must implement fit(X, y, sample_weight)
        and predict(X) methods.

    n_estimators : int
        Number of boosting iterations to perform.

    learning_rate : float, default=0.1
        Shrinkage parameter. Controls the contribution of each base learner.
        Lower values require more estimators but often result in better
        generalization.

    objective : Objective
        Loss function to optimize. Must be an instance of an Objective class
        that implements gradient() and loss() methods.

    second_order : bool, default=True
        If True, use second-order gradients (Newton boosting).
        If False, use only first-order gradients.

    subsample : float, default=1.0
        Fraction of samples to use for fitting each base learner.
        Must be in (0, 1]. Values < 1.0 enable stochastic gradient boosting.

    random_state : int or None, default=None
        Random seed for reproducibility of subsampling.

    callbacks : list of Callback, default=None
        List of callback instances for logging, early stopping, etc.

    Attributes
    ----------
    init_ : float
        Initial prediction value.

    _learners : list
        List of fitted base learners.

    stop_training : bool
        Flag set by callbacks to stop training early.

    Examples
    --------
    >>> from ggbm import GGBM
    >>> from ggbm.objectives import SquaredError
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> from sklearn.datasets import make_regression
    >>>
    >>> X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    >>> model = GGBM(
    ...     base_learner=DecisionTreeRegressor(max_depth=3),
    ...     n_estimators=50,
    ...     learning_rate=0.1,
    ...     objective=SquaredError()
    ... )
    >>> model.fit(X, y)
    >>> predictions = model.predict(X)
    """

    def __init__(self, base_learner, n_estimators, learning_rate=0.1,
                 objective: Objective=None, second_order=True, subsample=1.0,
                 random_state=None, callbacks: list[Callback]=None):
        
        assert subsample <=1.0 and subsample > 0.0, "subsample must be in (0.0, 1.0]"
        assert objective is not None, "objective must be provided"

        self.base_learner = base_learner
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.objective = objective
        self.second_order = second_order
        self.subsample = subsample
        self.random_state = random_state
        self.callbacks = callbacks if callbacks is not None else []
        self._learners = []
    
    def fit(self, X, y, sample_weight=None, X_val=None, y_val=None):
        """
        Fit the gradient boosting model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

        X_val : array-like of shape (n_val_samples, n_features), default=None
            Validation input samples (currently unused, reserved for future use).

        y_val : array-like of shape (n_val_samples,), default=None
            Validation target values (currently unused, reserved for future use).

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Reset learners if refitting
        if self._learners:
            self._learners = []
        self.stop_training = False

        rng = np.random.default_rng(self.random_state)
        obj = self.objective
        # Get scalar initial prediction
        init_value = obj.init_prediction(y, sample_weight=sample_weight)
        self.init_ = init_value
        # Create array of predictions initialized to this constant value
        y_pred = np.full_like(y, fill_value=init_value, dtype=np.float64)
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
        """
        Predict regression target for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y : array-like of shape (n_samples,)
            Predicted values.
        """
        base = getattr(self, "init_", 0.0)
        out = np.full(shape=(len(X),), fill_value=base, dtype=np.float64)
        
        for f in getattr(self, "_learners", []):
            out += self.learning_rate * f.predict(X)
        
        return out

    def _clone_base(self):
        """
        Clone the base learner.

        Returns
        -------
        estimator
            Deep copy of the base learner.
        """
        import copy
        return copy.deepcopy(self.base_learner)