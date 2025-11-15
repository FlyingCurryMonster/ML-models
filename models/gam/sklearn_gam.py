"""
Simple GAM using sklearn's SplineTransformer

A simple, efficient GAM implementation using sklearn's SplineTransformer
combined with Ridge regression. Much simpler and more reliable than statsmodels.
"""

import numpy as np
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline


class SklearnGAM:
    """
    Simple Generalized Additive Model using sklearn components.

    Combines SplineTransformer (for smooth nonlinear functions of each feature)
    with Ridge regression. This creates a GAM that's:
    - Simple and reliable
    - Fast to fit
    - sklearn-compatible
    - Perfect for boosting (adjustable complexity)

    Parameters
    ----------
    n_knots : int, default=5
        Number of knots for the spline basis.
        Fewer knots = simpler, smoother = better weak learner for boosting.

    degree : int, default=3
        Degree of the spline. 3 = cubic splines (smooth and flexible).

    alpha : float, default=1.0
        Ridge regularization parameter. Higher = simpler model.

    Examples
    --------
    >>> from models.gam import SklearnGAM
    >>> from ggbm import GGBM, SquaredError
    >>>
    >>> # Create a weak GAM learner
    >>> gam = SklearnGAM(n_knots=3, degree=2, alpha=10.0)
    >>>
    >>> # Boost it with GGBM
    >>> model = GGBM(
    ...     base_learner=gam,
    ...     n_estimators=50,
    ...     learning_rate=0.1,
    ...     objective=SquaredError()
    ... )
    """

    def __init__(self, n_knots=5, degree=3, alpha=1.0):
        self.n_knots = n_knots
        self.degree = degree
        self.alpha = alpha

        # Create the GAM pipeline:
        # 1. SplineTransformer: Creates smooth basis functions for each feature
        # 2. Ridge: Fits coefficients with regularization
        self.model_ = Pipeline([
            ('splines', SplineTransformer(
                n_knots=n_knots,
                degree=degree,
                include_bias=True
            )),
            ('ridge', Ridge(alpha=alpha, fit_intercept=False))
        ])

    def fit(self, X, y, sample_weight=None):
        """
        Fit the GAM model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights for Ridge regression.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Fit the pipeline
        if sample_weight is not None:
            # Pass sample weights to Ridge
            self.model_.fit(X, y, ridge__sample_weight=sample_weight)
        else:
            self.model_.fit(X, y)

        return self

    def predict(self, X):
        """
        Predict using the fitted GAM.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted values.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return self.model_.predict(X)

    def __repr__(self):
        return (
            f"SklearnGAM(n_knots={self.n_knots}, "
            f"degree={self.degree}, alpha={self.alpha})"
        )
