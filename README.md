# GGBM: Generalized Gradient Boosting for Any Model

A flexible gradient boosting framework that can boost **any** model architecture. Unlike traditional gradient boosting libraries that are tied to decision trees, GGBM allows you to boost linear models, neural networks, or any custom learner you can imagine.

## Features

- **Universal Boosting**: Boost any model that implements `fit()` and `predict()`
- **First & Second Order Gradients**: Support for both Newton boosting (second-order) and standard gradient boosting
- **Flexible Loss Functions**: Easy-to-extend objective system
- **sklearn Compatible**: Implements sklearn's estimator interface
- **Callbacks**: Early stopping, logging, and custom callbacks
- **Subsampling**: Stochastic gradient boosting support

## Installation

### From source (for development)

```bash
git clone https://github.com/yourusername/ggbm.git
cd ggbm
pip install -e .
```

### From PyPI (coming soon)

```bash
pip install ggbm
```

## Quick Start

### Basic Usage with Decision Trees

```python
from ggbm import GGBM
from ggbm.objectives import SquaredError
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a GGBM model that boosts decision trees
model = GGBM(
    base_learner=DecisionTreeRegressor(max_depth=3),
    n_estimators=100,
    learning_rate=0.1,
    objective=SquaredError(),
    second_order=True
)

# Fit and predict
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Boosting Linear Models

```python
from ggbm import GGBM
from ggbm.objectives import SquaredError
from sklearn.linear_model import Ridge

# Boost linear models!
model = GGBM(
    base_learner=Ridge(alpha=1.0),
    n_estimators=50,
    learning_rate=0.1,
    objective=SquaredError()
)

model.fit(X_train, y_train)
```

### Early Stopping

```python
from ggbm import GGBM, EarlyStopping
from ggbm.objectives import SquaredError
from sklearn.tree import DecisionTreeRegressor

# Create early stopping callback
early_stop = EarlyStopping(
    X_val=X_val,
    y_val=y_val,
    patience=10,
    min_delta=0.001,
    mode='min'
)

model = GGBM(
    base_learner=DecisionTreeRegressor(max_depth=3),
    n_estimators=1000,  # Will stop early
    learning_rate=0.1,
    objective=SquaredError(),
    callbacks=[early_stop]
)

model.fit(X_train, y_train)
```

### Stochastic Boosting

```python
# Use subsampling for faster training
model = GGBM(
    base_learner=DecisionTreeRegressor(max_depth=3),
    n_estimators=100,
    learning_rate=0.1,
    objective=SquaredError(),
    subsample=0.8,  # Use 80% of data per iteration
    random_state=42
)
```

## API Reference

### GGBM (GeneralizedGradBoostingRegressor)

**Parameters:**

- `base_learner`: The base model to boost. Must implement `fit(X, y, sample_weight)` and `predict(X)`
- `n_estimators`: Number of boosting iterations
- `learning_rate`: Shrinkage parameter (default: 0.1)
- `objective`: Loss function (must be an `Objective` instance)
- `second_order`: Use second-order gradients (Newton boosting) (default: True)
- `subsample`: Fraction of samples to use per iteration (default: 1.0)
- `random_state`: Random seed for reproducibility
- `callbacks`: List of callback instances for logging/early stopping

**Methods:**

- `fit(X, y, sample_weight=None)`: Fit the boosting model
- `predict(X)`: Make predictions

### Objectives

**SquaredError**: Mean squared error loss (L2)

```python
from ggbm.objectives import SquaredError
objective = SquaredError()
```

### Callbacks

**EarlyStopping**: Stop training when validation loss stops improving

```python
from ggbm import EarlyStopping
callback = EarlyStopping(X_val, y_val, patience=10, min_delta=0.001, mode='min')
```

## Creating Custom Objectives

```python
from ggbm.objectives import Objective
import numpy as np

class MyCustomLoss(Objective):
    def init_prediction(self, y, sample_weight=None):
        # Return initial prediction (e.g., mean for MSE)
        return np.full_like(y, fill_value=y.mean(), dtype=np.float64)

    def loss(self, y_true, y_pred, sample_weight=None):
        # Compute the loss
        return np.mean((y_true - y_pred) ** 2)

    def gradient(self, y_true, y_pred):
        # Return (gradient, hessian)
        grad = y_pred - y_true
        hess = np.ones_like(y_true)
        return grad, hess
```

## How It Works

GGBM implements gradient boosting using the following algorithm:

1. Initialize predictions with `objective.init_prediction(y)`
2. For each iteration:
   - Compute gradients and hessians: `grad, hess = objective.gradient(y, y_pred)`
   - Compute pseudo-residuals: `residuals = -grad / hess` (second-order) or `-grad` (first-order)
   - Fit base learner to pseudo-residuals with sample weights = hessian
   - Update predictions: `y_pred += learning_rate * learner.predict(X)`
3. Return ensemble of learners

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

## Roadmap

- [ ] Add classification support (GGBMClassifier)
- [ ] More built-in objectives (MAE, Huber, Quantile, LogLoss)
- [ ] Feature importance
- [ ] Model serialization helpers
- [ ] Verbose logging callback
- [ ] Parallel boosting
- [ ] GPU support

## Citation

If you use GGBM in your research, please cite:

```bibtex
@software{ggbm2024,
  author = {Your Name},
  title = {GGBM: Generalized Gradient Boosting for Any Model},
  year = {2024},
  url = {https://github.com/yourusername/ggbm}
}
```
