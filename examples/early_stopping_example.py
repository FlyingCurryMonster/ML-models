"""
Example demonstrating early stopping with GGBM
"""

from ggbm import GGBM, EarlyStopping
from ggbm.objectives import SquaredError
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate sample data
print("Generating sample data...")
X, y = make_regression(n_samples=1000, n_features=10, random_state=42, noise=10)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# Create early stopping callback
early_stop = EarlyStopping(
    X_val=X_val,
    y_val=y_val,
    patience=10,
    min_delta=0.1,
    mode='min'
)

# Train model with early stopping
print("\nTraining with early stopping...")
model = GGBM(
    base_learner=DecisionTreeRegressor(max_depth=3),
    n_estimators=500,  # Set high, will stop early
    learning_rate=0.1,
    objective=SquaredError(),
    callbacks=[early_stop],
    random_state=42
)

model.fit(X_train, y_train)

# Results
print(f"\n{'='*60}")
print("Early Stopping Results")
print(f"{'='*60}")
print(f"Requested estimators: 500")
print(f"Actual estimators trained: {len(model._learners)}")
print(f"Best iteration: {early_stop.best_iter}")
print(f"Best validation loss: {early_stop.best_score:.2f}")

# Evaluate on test set
y_pred = model.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred)
print(f"\nTest MSE: {test_mse:.2f}")

# Compare with model trained without early stopping
print(f"\n{'='*60}")
print("Without Early Stopping (for comparison)")
print(f"{'='*60}")

model_no_es = GGBM(
    base_learner=DecisionTreeRegressor(max_depth=3),
    n_estimators=len(model._learners),  # Train for same number of iterations
    learning_rate=0.1,
    objective=SquaredError(),
    random_state=42
)

model_no_es.fit(X_train, y_train)
y_pred_no_es = model_no_es.predict(X_test)
test_mse_no_es = mean_squared_error(y_test, y_pred_no_es)

print(f"Test MSE: {test_mse_no_es:.2f}")
print(f"Difference: {test_mse - test_mse_no_es:.2f}")
