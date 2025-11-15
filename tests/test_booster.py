import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from ggbm import GGBM, GeneralizedGradBoostingRegressor
from ggbm.objectives import SquaredError
from ggbm.callbacks import EarlyStopping


class TestGGBM:
    """Test suite for GeneralizedGradBoostingRegressor"""

    @pytest.fixture
    def regression_data(self):
        """Generate sample regression data"""
        X, y = make_regression(n_samples=200, n_features=10, random_state=42, noise=10)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def test_basic_fit_predict(self, regression_data):
        """Test basic fit and predict functionality"""
        X_train, X_test, y_train, y_test = regression_data

        model = GGBM(
            base_learner=DecisionTreeRegressor(max_depth=3),
            n_estimators=10,
            learning_rate=0.1,
            objective=SquaredError(),
            random_state=42
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        assert predictions.shape == (len(X_test),)
        assert not np.isnan(predictions).any()
        assert not np.isinf(predictions).any()

    def test_with_linear_model(self, regression_data):
        """Test boosting with linear models"""
        X_train, X_test, y_train, y_test = regression_data

        model = GGBM(
            base_learner=Ridge(alpha=1.0),
            n_estimators=20,
            learning_rate=0.1,
            objective=SquaredError()
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        assert predictions.shape == (len(X_test),)

    def test_second_order_vs_first_order(self, regression_data):
        """Test that second order boosting works"""
        X_train, X_test, y_train, y_test = regression_data

        model_first = GGBM(
            base_learner=DecisionTreeRegressor(max_depth=3),
            n_estimators=10,
            learning_rate=0.1,
            objective=SquaredError(),
            second_order=False,
            random_state=42
        )

        model_second = GGBM(
            base_learner=DecisionTreeRegressor(max_depth=3),
            n_estimators=10,
            learning_rate=0.1,
            objective=SquaredError(),
            second_order=True,
            random_state=42
        )

        model_first.fit(X_train, y_train)
        model_second.fit(X_train, y_train)

        pred_first = model_first.predict(X_test)
        pred_second = model_second.predict(X_test)

        # They should produce different predictions
        assert not np.allclose(pred_first, pred_second)

    def test_subsample(self, regression_data):
        """Test stochastic boosting with subsampling"""
        X_train, X_test, y_train, y_test = regression_data

        model = GGBM(
            base_learner=DecisionTreeRegressor(max_depth=3),
            n_estimators=10,
            learning_rate=0.1,
            objective=SquaredError(),
            subsample=0.8,
            random_state=42
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        assert predictions.shape == (len(X_test),)

    def test_early_stopping(self, regression_data):
        """Test early stopping callback"""
        X_train, X_test, y_train, y_test = regression_data

        # Split train into train and validation
        X_t, X_v, y_t, y_v = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        early_stop = EarlyStopping(
            X_val=X_v,
            y_val=y_v,
            patience=3,
            min_delta=0.0,
            mode='min'
        )

        model = GGBM(
            base_learner=DecisionTreeRegressor(max_depth=3),
            n_estimators=100,  # Should stop early
            learning_rate=0.1,
            objective=SquaredError(),
            callbacks=[early_stop],
            random_state=42
        )

        model.fit(X_t, y_t)

        # Should have stopped before 100 iterations
        assert len(model._learners) < 100

    def test_refit(self, regression_data):
        """Test refitting a model"""
        X_train, X_test, y_train, y_test = regression_data

        model = GGBM(
            base_learner=DecisionTreeRegressor(max_depth=3),
            n_estimators=5,
            learning_rate=0.1,
            objective=SquaredError(),
            random_state=42
        )

        model.fit(X_train, y_train)
        first_pred = model.predict(X_test)

        # Refit with different data
        model.fit(X_train[:100], y_train[:100])
        second_pred = model.predict(X_test)

        # Predictions should be different after refitting
        assert not np.allclose(first_pred, second_pred)
        # Should have reset learners
        assert len(model._learners) == 5

    def test_sklearn_compatibility(self, regression_data):
        """Test sklearn compatibility"""
        X_train, X_test, y_train, y_test = regression_data

        model = GGBM(
            base_learner=DecisionTreeRegressor(max_depth=3),
            n_estimators=10,
            learning_rate=0.1,
            objective=SquaredError()
        )

        # Should have sklearn methods
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'get_params')
        assert hasattr(model, 'set_params')

    def test_invalid_subsample(self):
        """Test that invalid subsample raises error"""
        with pytest.raises(AssertionError):
            GGBM(
                base_learner=DecisionTreeRegressor(),
                n_estimators=10,
                objective=SquaredError(),
                subsample=1.5  # Invalid
            )

    def test_missing_objective(self):
        """Test that missing objective raises error"""
        with pytest.raises(AssertionError):
            GGBM(
                base_learner=DecisionTreeRegressor(),
                n_estimators=10,
                objective=None  # Invalid
            )
