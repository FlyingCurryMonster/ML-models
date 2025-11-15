"""
Example demonstrating stochastic gradient boosting with subsampling
"""

from ggbm import GGBM
from ggbm.objectives import SquaredError
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time

# Generate sample data
print("Generating sample data...")
X, y = make_regression(n_samples=5000, n_features=20, random_state=42, noise=20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")

# Standard gradient boosting (subsample=1.0)
print("\n" + "="*60)
print("Standard Gradient Boosting (subsample=1.0)")
print("="*60)

start = time.time()
model_full = GGBM(
    base_learner=DecisionTreeRegressor(max_depth=3),
    n_estimators=100,
    learning_rate=0.1,
    objective=SquaredError(),
    subsample=1.0,
    random_state=42
)
model_full.fit(X_train, y_train)
time_full = time.time() - start

y_pred_full = model_full.predict(X_test)
mse_full = mean_squared_error(y_test, y_pred_full)

print(f"Training time: {time_full:.2f}s")
print(f"Test MSE: {mse_full:.2f}")

# Stochastic gradient boosting (subsample=0.8)
print("\n" + "="*60)
print("Stochastic Gradient Boosting (subsample=0.8)")
print("="*60)

start = time.time()
model_stochastic = GGBM(
    base_learner=DecisionTreeRegressor(max_depth=3),
    n_estimators=100,
    learning_rate=0.1,
    objective=SquaredError(),
    subsample=0.8,
    random_state=42
)
model_stochastic.fit(X_train, y_train)
time_stochastic = time.time() - start

y_pred_stochastic = model_stochastic.predict(X_test)
mse_stochastic = mean_squared_error(y_test, y_pred_stochastic)

print(f"Training time: {time_stochastic:.2f}s")
print(f"Test MSE: {mse_stochastic:.2f}")

# Stochastic gradient boosting (subsample=0.5)
print("\n" + "="*60)
print("Stochastic Gradient Boosting (subsample=0.5)")
print("="*60)

start = time.time()
model_very_stochastic = GGBM(
    base_learner=DecisionTreeRegressor(max_depth=3),
    n_estimators=100,
    learning_rate=0.1,
    objective=SquaredError(),
    subsample=0.5,
    random_state=42
)
model_very_stochastic.fit(X_train, y_train)
time_very_stochastic = time.time() - start

y_pred_very_stochastic = model_very_stochastic.predict(X_test)
mse_very_stochastic = mean_squared_error(y_test, y_pred_very_stochastic)

print(f"Training time: {time_very_stochastic:.2f}s")
print(f"Test MSE: {mse_very_stochastic:.2f}")

# Summary
print("\n" + "="*60)
print("Summary")
print("="*60)
print(f"{'Subsample':<15} {'Time (s)':<12} {'MSE':<12} {'Speedup':<10}")
print("-"*60)
print(f"{1.0:<15} {time_full:<12.2f} {mse_full:<12.2f} {1.0:<10.2f}x")
print(f"{0.8:<15} {time_stochastic:<12.2f} {mse_stochastic:<12.2f} {time_full/time_stochastic:<10.2f}x")
print(f"{0.5:<15} {time_very_stochastic:<12.2f} {mse_very_stochastic:<12.2f} {time_full/time_very_stochastic:<10.2f}x")
