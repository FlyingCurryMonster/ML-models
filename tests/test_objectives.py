import pytest
import numpy as np
from ggbm.objectives import SquaredError


class TestSquaredError:
    """Test suite for SquaredError objective"""

    def test_init_prediction(self):
        """Test initial prediction returns mean as a scalar"""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        obj = SquaredError()
        init_pred = obj.init_prediction(y)

        # Should return a scalar, not an array
        assert isinstance(init_pred, (float, np.floating))
        assert np.isclose(init_pred, 3.0)  # Mean of y

    def test_init_prediction_with_weights(self):
        """Test initial prediction with sample weights"""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.array([1.0, 1.0, 1.0, 1.0, 10.0])
        obj = SquaredError()
        init_pred = obj.init_prediction(y, sample_weight=weights)

        # Should return a scalar
        assert isinstance(init_pred, (float, np.floating))
        # Weighted mean should be closer to 5.0
        assert init_pred > 3.0

    def test_loss(self):
        """Test loss computation"""
        obj = SquaredError()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])

        loss = obj.loss(y_true, y_pred)
        expected_loss = np.mean((y_true - y_pred) ** 2)

        assert np.isclose(loss, expected_loss)
        assert np.isclose(loss, 0.25)

    def test_gradient(self):
        """Test gradient and hessian computation"""
        obj = SquaredError()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])

        grad, hess = obj.gradient(y_true, y_pred)

        expected_grad = y_pred - y_true
        expected_hess = np.ones_like(y_true)

        assert np.allclose(grad, expected_grad)
        assert np.allclose(hess, expected_hess)

    def test_gradient_shapes(self):
        """Test that gradient and hessian have correct shapes"""
        obj = SquaredError()
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        grad, hess = obj.gradient(y_true, y_pred)

        assert grad.shape == y_true.shape
        assert hess.shape == y_true.shape
