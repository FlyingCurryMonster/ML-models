import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

from ggbm import GGBM
from ggbm.objectives import SquaredError
from ggbm.callbacks import EarlyStopping


class TestEarlyStopping:
    """Test suite for EarlyStopping callback"""

    @pytest.fixture
    def regression_data(self):
        """Generate sample regression data"""
        X, y = make_regression(n_samples=200, n_features=10, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return X_train, X_val, y_train, y_val

    def test_early_stopping_min_mode(self, regression_data):
        """Test early stopping in minimize mode"""
        X_train, X_val, y_train, y_val = regression_data

        early_stop = EarlyStopping(
            X_val=X_val,
            y_val=y_val,
            patience=5,
            min_delta=0.0,
            mode='min'
        )

        model = GGBM(
            base_learner=DecisionTreeRegressor(max_depth=3),
            n_estimators=100,
            learning_rate=0.1,
            objective=SquaredError(),
            callbacks=[early_stop],
            random_state=42
        )

        model.fit(X_train, y_train)

        # Early stopping should work (may or may not trigger depending on data)
        # Best iteration should be recorded
        assert early_stop.best_iter >= 0
        # Model should have some learners
        assert len(model._learners) > 0
        assert len(model._learners) <= 100

    def test_early_stopping_patience(self, regression_data):
        """Test that patience parameter works correctly"""
        X_train, X_val, y_train, y_val = regression_data

        # Very low patience - should stop quickly
        early_stop = EarlyStopping(
            X_val=X_val,
            y_val=y_val,
            patience=2,
            min_delta=0.0,
            mode='min'
        )

        model = GGBM(
            base_learner=DecisionTreeRegressor(max_depth=3),
            n_estimators=100,
            learning_rate=0.1,
            objective=SquaredError(),
            callbacks=[early_stop],
            random_state=42
        )

        model.fit(X_train, y_train)

        # Should stop before reaching max iterations
        # (actual number depends on data, but should stop at some point)
        assert len(model._learners) < 100
        assert len(model._learners) > 0

    def test_multiple_callbacks(self, regression_data):
        """Test using multiple callbacks"""
        X_train, X_val, y_train, y_val = regression_data

        early_stop1 = EarlyStopping(
            X_val=X_val,
            y_val=y_val,
            patience=10,
            min_delta=0.0,
            mode='min'
        )

        early_stop2 = EarlyStopping(
            X_val=X_val,
            y_val=y_val,
            patience=5,
            min_delta=0.0,
            mode='min'
        )

        model = GGBM(
            base_learner=DecisionTreeRegressor(max_depth=3),
            n_estimators=100,
            learning_rate=0.1,
            objective=SquaredError(),
            callbacks=[early_stop1, early_stop2],
            random_state=42
        )

        model.fit(X_train, y_train)

        # Should work with multiple callbacks without errors
        assert len(model._learners) > 0
        assert len(model._learners) <= 100
        # Both callbacks should have been executed
        assert early_stop1.best_iter >= 0
        assert early_stop2.best_iter >= 0
