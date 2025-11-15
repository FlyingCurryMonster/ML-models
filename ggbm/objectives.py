# objectives.py
import numpy as np

class Objective:
    def init_prediction(self, y, sample_weight=None):
        """
        The initial prediction given a loss function.
        e.g. MSE loss predicts the mean of the response
        and MAE loss predicts the median of the response
        """
        raise NotImplementedError
    def loss(self, y_true, y_pred) -> float:    
        """
        Compute the loss given true and predicted values
        """
        raise NotImplementedError

    def gradient(self, y_true, y_pred):
        """
        Return gradient and hessian as tuple (grad, hess)
        """
        raise NotImplementedError

class SquaredError(Objective):
    """Mean Squared Error (L2 loss)"""

    def init_prediction(self, y, sample_weight=None):
        if sample_weight is not None:
            base = np.average(y, weights=sample_weight)
        else:
            base = y.mean()
        return np.full_like(y, fill_value=base, dtype=np.float64)

    def loss(self, y_true, y_pred, sample_weight=None) -> float:
        if sample_weight is not None:
            return np.average((y_true - y_pred) ** 2, weights=sample_weight)
        else:
            return np.mean((y_true - y_pred) ** 2)

    def gradient(self, y_true, y_pred):
        grad = y_pred - y_true
        hess = np.ones_like(y_true)
        return grad, hess


class AbsoluteError(Objective):
    """Mean Absolute Error (L1 loss)"""

    def init_prediction(self, y, sample_weight=None):
        if sample_weight is not None:
            # Weighted median is complex, use weighted mean as approximation
            base = np.average(y, weights=sample_weight)
        else:
            base = np.median(y)
        return np.full_like(y, fill_value=base, dtype=np.float64)

    def loss(self, y_true, y_pred, sample_weight=None) -> float:
        if sample_weight is not None:
            return np.average(np.abs(y_true - y_pred), weights=sample_weight)
        else:
            return np.mean(np.abs(y_true - y_pred))

    def gradient(self, y_true, y_pred):
        residual = y_pred - y_true
        # Gradient of |x| is sign(x), but undefined at 0
        # Use small epsilon to avoid division by zero
        grad = np.sign(residual)
        # Hessian is technically 0 everywhere except at 0 where it's undefined
        # Use small constant for numerical stability
        hess = np.full_like(y_true, fill_value=1e-6)
        return grad, hess


class HuberLoss(Objective):
    """
    Huber loss - combines MSE and MAE
    Quadratic for small errors, linear for large errors
    """

    def __init__(self, delta=1.0):
        """
        Parameters:
        -----------
        delta : float
            Threshold at which to switch from quadratic to linear loss
        """
        self.delta = delta

    def init_prediction(self, y, sample_weight=None):
        if sample_weight is not None:
            base = np.average(y, weights=sample_weight)
        else:
            base = y.mean()
        return np.full_like(y, fill_value=base, dtype=np.float64)

    def loss(self, y_true, y_pred, sample_weight=None) -> float:
        residual = y_true - y_pred
        abs_residual = np.abs(residual)

        # Huber loss formula
        loss = np.where(
            abs_residual <= self.delta,
            0.5 * residual ** 2,
            self.delta * (abs_residual - 0.5 * self.delta)
        )

        if sample_weight is not None:
            return np.average(loss, weights=sample_weight)
        else:
            return np.mean(loss)

    def gradient(self, y_true, y_pred):
        residual = y_pred - y_true
        abs_residual = np.abs(residual)

        # Gradient
        grad = np.where(
            abs_residual <= self.delta,
            residual,
            self.delta * np.sign(residual)
        )

        # Hessian
        hess = np.where(
            abs_residual <= self.delta,
            1.0,
            1e-6  # Small constant for linear region
        )

        return grad, hess


class QuantileLoss(Objective):
    """
    Quantile loss for quantile regression
    """

    def __init__(self, alpha=0.5):
        """
        Parameters:
        -----------
        alpha : float
            Quantile level (0 < alpha < 1)
            alpha=0.5 is median regression
        """
        assert 0 < alpha < 1, "alpha must be in (0, 1)"
        self.alpha = alpha

    def init_prediction(self, y, sample_weight=None):
        if sample_weight is None:
            base = np.quantile(y, self.alpha)
        else:
            # Weighted quantile is complex, use mean as approximation
            base = np.average(y, weights=sample_weight)
        return np.full_like(y, fill_value=base, dtype=np.float64)

    def loss(self, y_true, y_pred, sample_weight=None) -> float:
        residual = y_true - y_pred
        loss = np.where(
            residual >= 0,
            self.alpha * residual,
            (self.alpha - 1) * residual
        )

        if sample_weight is not None:
            return np.average(loss, weights=sample_weight)
        else:
            return np.mean(loss)

    def gradient(self, y_true, y_pred):
        residual = y_pred - y_true

        # Gradient
        grad = np.where(
            residual >= 0,
            self.alpha,
            self.alpha - 1
        )

        # Hessian is 0 (piecewise linear), use small constant
        hess = np.full_like(y_true, fill_value=1e-6)

        return grad, hess