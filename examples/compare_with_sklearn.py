"""
Comprehensive comparison of GGBM with sklearn models

Compares:
1. Single Decision Tree (baseline)
2. sklearn GradientBoostingRegressor (reference implementation)
3. GGBM with Decision Trees (our implementation)

Shows that GGBM produces similar results to sklearn's gradient boosting.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from ggbm import GGBM
from ggbm.objectives import SquaredError

# Set random seed for reproducibility
np.random.seed(42)

print("="*70)
print("GGBM vs sklearn Gradient Boosting Comparison")
print("="*70)

# Load dataset
print("\nLoading California Housing dataset...")
data = fetch_california_housing()
X, y = data.data, data.target

print(f"Dataset shape: {X.shape}")
print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Model parameters (keep them similar for fair comparison)
N_ESTIMATORS = 100
MAX_DEPTH = 3
LEARNING_RATE = 0.1

print(f"\nHyperparameters:")
print(f"  n_estimators: {N_ESTIMATORS}")
print(f"  max_depth: {MAX_DEPTH}")
print(f"  learning_rate: {LEARNING_RATE}")

# ============================================================================
# Model 1: Single Decision Tree (baseline)
# ============================================================================
print("\n" + "="*70)
print("Training Single Decision Tree (baseline)...")
print("="*70)

dt_model = DecisionTreeRegressor(max_depth=MAX_DEPTH, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

dt_mse = mean_squared_error(y_test, y_pred_dt)
dt_r2 = r2_score(y_test, y_pred_dt)
dt_mae = mean_absolute_error(y_test, y_pred_dt)

print(f"Decision Tree - MSE: {dt_mse:.4f}, R²: {dt_r2:.4f}, MAE: {dt_mae:.4f}")

# ============================================================================
# Model 2: sklearn GradientBoostingRegressor
# ============================================================================
print("\n" + "="*70)
print("Training sklearn GradientBoostingRegressor...")
print("="*70)

sklearn_gbm = GradientBoostingRegressor(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    learning_rate=LEARNING_RATE,
    random_state=42
)
sklearn_gbm.fit(X_train, y_train)
y_pred_sklearn = sklearn_gbm.predict(X_test)

sklearn_mse = mean_squared_error(y_test, y_pred_sklearn)
sklearn_r2 = r2_score(y_test, y_pred_sklearn)
sklearn_mae = mean_absolute_error(y_test, y_pred_sklearn)

print(f"sklearn GBM - MSE: {sklearn_mse:.4f}, R²: {sklearn_r2:.4f}, MAE: {sklearn_mae:.4f}")

# ============================================================================
# Model 3: GGBM with Decision Trees
# ============================================================================
print("\n" + "="*70)
print("Training GGBM...")
print("="*70)

ggbm_model = GGBM(
    base_learner=DecisionTreeRegressor(max_depth=MAX_DEPTH),
    n_estimators=N_ESTIMATORS,
    learning_rate=LEARNING_RATE,
    objective=SquaredError(),
    random_state=42
)
ggbm_model.fit(X_train, y_train)
y_pred_ggbm = ggbm_model.predict(X_test)

ggbm_mse = mean_squared_error(y_test, y_pred_ggbm)
ggbm_r2 = r2_score(y_test, y_pred_ggbm)
ggbm_mae = mean_absolute_error(y_test, y_pred_ggbm)

print(f"GGBM - MSE: {ggbm_mse:.4f}, R²: {ggbm_r2:.4f}, MAE: {ggbm_mae:.4f}")

# ============================================================================
# Comparison Summary
# ============================================================================
print("\n" + "="*70)
print("COMPARISON SUMMARY")
print("="*70)
print(f"\n{'Model':<30} {'MSE':<12} {'R²':<12} {'MAE':<12}")
print("-"*70)
print(f"{'Single Decision Tree':<30} {dt_mse:<12.4f} {dt_r2:<12.4f} {dt_mae:<12.4f}")
print(f"{'sklearn GradientBoosting':<30} {sklearn_mse:<12.4f} {sklearn_r2:<12.4f} {sklearn_mae:<12.4f}")
print(f"{'GGBM':<30} {ggbm_mse:<12.4f} {ggbm_r2:<12.4f} {ggbm_mae:<12.4f}")

# Compute difference between GGBM and sklearn
mse_diff = abs(ggbm_mse - sklearn_mse) / sklearn_mse * 100
r2_diff = abs(ggbm_r2 - sklearn_r2) / abs(sklearn_r2) * 100
mae_diff = abs(ggbm_mae - sklearn_mae) / sklearn_mae * 100

print("\n" + "="*70)
print("GGBM vs sklearn GBM (% difference)")
print("="*70)
print(f"MSE difference: {mse_diff:.2f}%")
print(f"R² difference:  {r2_diff:.2f}%")
print(f"MAE difference: {mae_diff:.2f}%")

# Correlation between predictions
corr = np.corrcoef(y_pred_sklearn, y_pred_ggbm)[0, 1]
print(f"\nCorrelation between sklearn and GGBM predictions: {corr:.6f}")

# ============================================================================
# Visualizations
# ============================================================================
print("\n" + "="*70)
print("Creating visualizations...")
print("="*70)

fig = plt.figure(figsize=(16, 10))

# Plot 1: Decision Tree - y_pred vs y_true
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(y_test, y_pred_dt, alpha=0.5, s=10)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel('True Values', fontsize=10)
ax1.set_ylabel('Predictions', fontsize=10)
ax1.set_title(f'Decision Tree\nMSE={dt_mse:.4f}, R²={dt_r2:.4f}', fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: sklearn GBM - y_pred vs y_true
ax2 = plt.subplot(2, 3, 2)
ax2.scatter(y_test, y_pred_sklearn, alpha=0.5, s=10, color='green')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_xlabel('True Values', fontsize=10)
ax2.set_ylabel('Predictions', fontsize=10)
ax2.set_title(f'sklearn GradientBoosting\nMSE={sklearn_mse:.4f}, R²={sklearn_r2:.4f}', fontsize=11)
ax2.grid(True, alpha=0.3)

# Plot 3: GGBM - y_pred vs y_true
ax3 = plt.subplot(2, 3, 3)
ax3.scatter(y_test, y_pred_ggbm, alpha=0.5, s=10, color='purple')
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax3.set_xlabel('True Values', fontsize=10)
ax3.set_ylabel('Predictions', fontsize=10)
ax3.set_title(f'GGBM\nMSE={ggbm_mse:.4f}, R²={ggbm_r2:.4f}', fontsize=11)
ax3.grid(True, alpha=0.3)

# Plot 4: GGBM vs sklearn GBM predictions (should be close to y=x)
ax4 = plt.subplot(2, 3, 4)
ax4.scatter(y_pred_sklearn, y_pred_ggbm, alpha=0.5, s=10, color='orange')
ax4.plot([y_pred_sklearn.min(), y_pred_sklearn.max()],
         [y_pred_sklearn.min(), y_pred_sklearn.max()], 'r--', lw=2)
ax4.set_xlabel('sklearn GBM Predictions', fontsize=10)
ax4.set_ylabel('GGBM Predictions', fontsize=10)
ax4.set_title(f'GGBM vs sklearn GBM\nCorrelation={corr:.6f}', fontsize=11)
ax4.grid(True, alpha=0.3)

# Plot 5: Residuals comparison
ax5 = plt.subplot(2, 3, 5)
residuals_sklearn = y_test - y_pred_sklearn
residuals_ggbm = y_test - y_pred_ggbm
ax5.scatter(y_pred_sklearn, residuals_sklearn, alpha=0.5, s=10, label='sklearn GBM', color='green')
ax5.scatter(y_pred_ggbm, residuals_ggbm, alpha=0.5, s=10, label='GGBM', color='purple')
ax5.axhline(y=0, color='r', linestyle='--', lw=2)
ax5.set_xlabel('Predictions', fontsize=10)
ax5.set_ylabel('Residuals (True - Predicted)', fontsize=10)
ax5.set_title('Residual Plot', fontsize=11)
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Prediction difference histogram
ax6 = plt.subplot(2, 3, 6)
pred_diff = y_pred_ggbm - y_pred_sklearn
ax6.hist(pred_diff, bins=50, edgecolor='black', alpha=0.7, color='cyan')
ax6.axvline(x=0, color='r', linestyle='--', lw=2)
ax6.set_xlabel('Prediction Difference (GGBM - sklearn)', fontsize=10)
ax6.set_ylabel('Frequency', fontsize=10)
ax6.set_title(f'Prediction Differences\nMean={pred_diff.mean():.6f}, Std={pred_diff.std():.6f}', fontsize=11)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('examples/ggbm_sklearn_comparison.png', dpi=150, bbox_inches='tight')
print("Plot saved to: examples/ggbm_sklearn_comparison.png")
plt.show()

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
if mse_diff < 5 and corr > 0.99:
    print("✓ GGBM predictions are very close to sklearn's GradientBoosting!")
    print("✓ The implementation is working correctly.")
elif mse_diff < 10 and corr > 0.95:
    print("✓ GGBM predictions are similar to sklearn's GradientBoosting.")
    print("  Minor differences may be due to implementation details.")
else:
    print("⚠ GGBM predictions differ from sklearn's GradientBoosting.")
    print("  This may indicate an issue or different implementation choices.")

print("\nNote: Both gradient boosting methods should significantly outperform")
print("the single decision tree baseline.")
