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