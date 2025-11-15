"""
Example of boosting linear models with GGBM
"""

from ggbm import GGBM
from ggbm.objectives import SquaredError
from sklearn.linear_model import Ridge, Lasso
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data
print("Generating sample data...")
X, y = make_regression(n_samples=1000, n_features=20, random_state=42, noise=20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Boost Ridge regression
print("\n" + "="*50)
print("Boosting Ridge Regression")
print("="*50)

model_ridge = GGBM(
    base_learner=Ridge(alpha=1.0),
    n_estimators=50,
    learning_rate=0.1,
    objective=SquaredError(),
    random_state=42
)

model_ridge.fit(X_train, y_train)
y_pred_ridge = model_ridge.predict(X_test)

print(f"MSE: {mean_squared_error(y_test, y_pred_ridge):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred_ridge):.4f}")

# Boost Lasso regression
print("\n" + "="*50)
print("Boosting Lasso Regression")
print("="*50)

model_lasso = GGBM(
    base_learner=Lasso(alpha=0.1),
    n_estimators=50,
    learning_rate=0.1,
    objective=SquaredError(),
    random_state=42
)

model_lasso.fit(X_train, y_train)
y_pred_lasso = model_lasso.predict(X_test)

print(f"MSE: {mean_squared_error(y_test, y_pred_lasso):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred_lasso):.4f}")

# Compare with baseline linear model
print("\n" + "="*50)
print("Baseline Ridge (no boosting)")
print("="*50)

baseline = Ridge(alpha=1.0)
baseline.fit(X_train, y_train)
y_pred_baseline = baseline.predict(X_test)

print(f"MSE: {mean_squared_error(y_test, y_pred_baseline):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred_baseline):.4f}")
