"""
Basic usage example of GGBM with decision trees
"""

from ggbm import GGBM
from ggbm.objectives import SquaredError
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data
print("Generating sample data...")
X, y = make_regression(n_samples=1000, n_features=10, random_state=42, noise=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train GGBM model
print("\nTraining GGBM model...")
model = GGBM(
    base_learner=DecisionTreeRegressor(max_depth=3),
    n_estimators=100,
    learning_rate=0.1,
    objective=SquaredError(),
    second_order=True,
    random_state=42
)

model.fit(X_train, y_train)

# Make predictions
print("\nMaking predictions...")
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nResults:")
print(f"  MSE: {mse:.2f}")
print(f"  R² Score: {r2:.4f}")
print(f"  Number of learners: {len(model._learners)}")
