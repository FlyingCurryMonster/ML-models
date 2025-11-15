"""
Statsmodels GAM Wrapper

A scikit-learn-compatible wrapper around statsmodels' Generalized Additive Models.
Designed to work with GGBM for boosting GAMs.
"""

import numpy as np
from statsmodels.gam.api import GLMGam, BSplines


class StatsmodelsGAM:
    """
    Wrapper for statsmodels GLMGam that provides sklearn-like interface.

    This wrapper makes statsmodels GAMs compatible with GGBM and provides
    a familiar fit/predict API.

    Parameters
    ----------
    n_splines : int or list of int, default=10
        Number of basis splines for each feature.
        If int, same number used for all features.
        If list, must match number of features.

    degree : int, default=3
        Degree of the spline basis functions.

    alpha : float, default=0.0
        Regularization parameter for the GAM smoothing penalty.

    family : statsmodels family object, default=None
        GLM family (e.g., Gaussian(), Poisson()).
        If None, uses Gaussian() for regression.

    Examples
    --------
    >>> from models.gam import StatsmodelsGAM
    >>> from ggbm import GGBM, SquaredError
    >>>
    >>> # Create a weak GAM learner
    >>> gam = StatsmodelsGAM(n_splines=5, degree=2)
    >>>
    >>> # Boost it with GGBM
    >>> model = GGBM(
    ...     base_learner=gam,
    ...     n_estimators=50,
    ...     learning_rate=0.1,
    ...     objective=SquaredError()
    ... )
    """

    def __init__(self, n_splines=10, degree=3, alpha=0.0, family=None):
        self.n_splines = n_splines
        self.degree = degree
        self.alpha = alpha
        self.family = family
        self.model_ = None
        self.result_ = None
        self.bs_ = None  # Store the spline transformer
        self.n_features_in_ = None

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
            Sample weights. Currently not supported by statsmodels GAM.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_features_in_ = X.shape[1]

        # Set up spline basis for each feature
        if isinstance(self.n_splines, int):
            n_splines_list = [self.n_splines] * self.n_features_in_
        else:
            n_splines_list = self.n_splines
            if len(n_splines_list) != self.n_features_in_:
                raise ValueError(
                    f"Length of n_splines ({len(n_splines_list)}) must match "
                    f"number of features ({self.n_features_in_})"
                )

        # Create alpha list
        alpha = [self.alpha] * self.n_features_in_

        # Set family (default to Gaussian for regression)
        if self.family is None:
            from statsmodels.genmod.families import Gaussian
            family = Gaussian()
        else:
            family = self.family

        # Create B-spline basis
        # df parameter specifies degrees of freedom for each feature
        # degree specifies the polynomial degree
        self.bs_ = BSplines(
            X,
            df=n_splines_list,
            degree=[self.degree] * self.n_features_in_
        )

        # Fit the GAM
        self.model_ = GLMGam(y, smoother=self.bs_, alpha=alpha, family=family)
        self.result_ = self.model_.fit()

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
        if self.result_ is None:
            raise ValueError("Model must be fitted before calling predict()")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was fitted with "
                f"{self.n_features_in_} features"
            )

        # Transform X using the same spline basis
        # Create a new BSplines with the same configuration
        bs_pred = BSplines(
            X,
            df=[self.bs_.df[i] for i in range(self.n_features_in_)],
            degree=[self.degree] * self.n_features_in_
        )

        # Get predictions using the transformed features
        # We need to manually transform and predict
        X_transformed = bs_pred.smoothers[0].basis
        for i in range(1, self.n_features_in_):
            X_transformed = np.column_stack([X_transformed, bs_pred.smoothers[i].basis])

        # Use the fitted parameters to make predictions
        # Predict using linear predictor
        pred = self.result_.predict(exog=X)

        return pred

    def __repr__(self):
        return (
            f"StatsmodelsGAM(n_splines={self.n_splines}, "
            f"degree={self.degree}, alpha={self.alpha})"
        )
