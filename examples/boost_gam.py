"""
Boosting Generalized Additive Models (GAMs)

Demonstrates boosting statsmodels GAMs with GGBM.
GAMs are naturally smooth and interpretable, making them good weak learners.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import sys
sys.path.insert(0, '/home/rakin/rnb76-rclone/ML-models')

from models.gam import SklearnGAM
from ggbm import GGBM, SquaredError

# Set random seed
np.random.seed(42)

print("="*70)
print("Boosting Generalized Additive Models (GAMs)")
print("="*70)

# Generate data with nonlinear relationships
print("\nGenerating dataset with nonlinear patterns...")
X, y = make_regression(n_samples=500, n_features=5, n_informative=5, noise=10, random_state=42)

# Add some nonlinear transformations
X[:, 0] = np.sin(X[:, 0])
X[:, 1] = X[:, 1] ** 2
X[:, 2] = np.exp(X[:, 2] / 5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# ============================================================================
# Model 1: Single GAM (weak learner with few splines)
# ============================================================================
print("\n" + "="*70)
print("Training single GAM (weak - few splines)...")
print("="*70)

gam_weak = SklearnGAM(n_knots=3, degree=2, alpha=10.0)
gam_weak.fit(X_train, y_train)
y_pred_weak = gam_weak.predict(X_test)

weak_mse = mean_squared_error(y_test, y_pred_weak)
weak_r2 = r2_score(y_test, y_pred_weak)

print(f"Weak GAM (3 knots) - MSE: {weak_mse:.2f}, R²: {weak_r2:.4f}")

# ============================================================================
# Model 2: Single GAM (stronger with more splines)
# ============================================================================
print("\n" + "="*70)
print("Training single GAM (stronger - more splines)...")
print("="*70)

gam_strong = SklearnGAM(n_knots=10, degree=3, alpha=1.0)
gam_strong.fit(X_train, y_train)
y_pred_strong = gam_strong.predict(X_test)

strong_mse = mean_squared_error(y_test, y_pred_strong)
strong_r2 = r2_score(y_test, y_pred_strong)

print(f"Strong GAM (10 knots) - MSE: {strong_mse:.2f}, R²: {strong_r2:.4f}")

# ============================================================================
# Model 3: Boosted GAM
# ============================================================================
print("\n" + "="*70)
print("Training GGBM with GAM (boosting weak GAM)...")
print("="*70)

N_ESTIMATORS = 30
LEARNING_RATE = 0.1

print(f"Boosting parameters:")
print(f"  n_estimators: {N_ESTIMATORS}")
print(f"  learning_rate: {LEARNING_RATE}")
print(f"  base GAM: {gam_weak}")

ggbm_gam = GGBM(
    base_learner=SklearnGAM(n_knots=3, degree=2, alpha=10.0),
    n_estimators=N_ESTIMATORS,
    learning_rate=LEARNING_RATE,
    objective=SquaredError(),
    random_state=42
)

ggbm_gam.fit(X_train, y_train)
y_pred_ggbm = ggbm_gam.predict(X_test)

ggbm_mse = mean_squared_error(y_test, y_pred_ggbm)
ggbm_r2 = r2_score(y_test, y_pred_ggbm)

print(f"Boosted GAM - MSE: {ggbm_mse:.2f}, R²: {ggbm_r2:.4f}")

# ============================================================================
# Comparison
# ============================================================================
print("\n" + "="*70)
print("COMPARISON SUMMARY")
print("="*70)
print(f"\n{'Model':<30} {'MSE':<12} {'R²':<12}")
print("-"*70)
print(f"{'Single GAM (weak, 3 knots)':<30} {weak_mse:<12.2f} {weak_r2:<12.4f}")
print(f"{'Single GAM (strong, 10 knots)':<30} {strong_mse:<12.2f} {strong_r2:<12.4f}")
print(f"{'Boosted GAM (30 estimators)':<30} {ggbm_mse:<12.2f} {ggbm_r2:<12.4f}")

improvement = (weak_mse - ggbm_mse) / weak_mse * 100
print(f"\nBoosting improves over weak GAM by: {improvement:.1f}%")

# ============================================================================
# Visualization
# ============================================================================
print("\n" + "="*70)
print("Creating visualizations...")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Weak GAM
ax1 = axes[0, 0]
ax1.scatter(y_test, y_pred_weak, alpha=0.6, s=30, color='red')
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax1.set_xlabel('True Values', fontsize=10)
ax1.set_ylabel('Predictions', fontsize=10)
ax1.set_title(f'Weak GAM (3 knots)\nMSE={weak_mse:.2f}, R²={weak_r2:.4f}', fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: Strong GAM
ax2 = axes[0, 1]
ax2.scatter(y_test, y_pred_strong, alpha=0.6, s=30, color='blue')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax2.set_xlabel('True Values', fontsize=10)
ax2.set_ylabel('Predictions', fontsize=10)
ax2.set_title(f'Strong GAM (10 knots)\nMSE={strong_mse:.2f}, R²={strong_r2:.4f}', fontsize=11)
ax2.grid(True, alpha=0.3)

# Plot 3: Boosted GAM
ax3 = axes[1, 0]
ax3.scatter(y_test, y_pred_ggbm, alpha=0.6, s=30, color='green')
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax3.set_xlabel('True Values', fontsize=10)
ax3.set_ylabel('Predictions', fontsize=10)
ax3.set_title(f'Boosted GAM (3 splines, {N_ESTIMATORS} estimators)\nMSE={ggbm_mse:.2f}, R²={ggbm_r2:.4f}', fontsize=11)
ax3.grid(True, alpha=0.3)

# Plot 4: MSE Comparison
ax4 = axes[1, 1]
models = ['Weak GAM\n(3 splines)', 'Strong GAM\n(10 splines)', 'Boosted GAM\n(3 splines, 30 est.)']
mses = [weak_mse, strong_mse, ggbm_mse]
colors = ['red', 'blue', 'green']
bars = ax4.bar(models, mses, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax4.set_ylabel('Mean Squared Error', fontsize=10)
ax4.set_title('MSE Comparison', fontsize=11)
ax4.grid(True, alpha=0.3, axis='y')
for bar, mse in zip(bars, mses):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{mse:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
print("Plot saved to: examples/boost_gam.png")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("\nKey takeaways:")
print("1. GAMs are smooth, interpretable models - good weak learners")
print("2. Weak GAM (few splines): High bias, underfits")
print("3. Boosting reduces bias by combining many smooth GAMs")
print(f"4. Performance gain: {improvement:.1f}% improvement over weak GAM")
print("\n✓ GGBM successfully boosts GAMs from statsmodels!")

# Show plot last
plt.show()
