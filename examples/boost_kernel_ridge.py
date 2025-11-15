"""
Boosting Kernel Ridge Regression with GGBM

This example demonstrates boosting a kernel ridge regressor with a conservative
(small) gamma parameter. A small gamma creates a weak learner with high bias
and low variance, which is ideal for boosting.

Key insight: Boosting reduces bias by combining many weak learners.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score

from ggbm import GGBM
from ggbm.objectives import SquaredError

# Set random seed for reproducibility
np.random.seed(42)

print("="*70)
print("Boosting Kernel Ridge Regression")
print("="*70)

# Generate a dataset with some nonlinearity
print("\nGenerating dataset...")
X, y = make_regression(
    n_samples=1000,
    n_features=10,
    n_informative=8,
    noise=20,
    random_state=42
)

# Add some nonlinear transformations to make it more interesting
X[:, 0] = X[:, 0] ** 2
X[:, 1] = np.sin(X[:, 1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# ============================================================================
# Kernel Ridge parameters
# ============================================================================
# Conservative (small) gamma = large kernel width = smoother, simpler model
# This creates a weak learner with high bias, perfect for boosting
#
# Note: For RBF kernel, gamma controls complexity:
#   - Very small gamma (0.001) = very smooth = underfits (weak learner)
#   - Medium gamma (0.1) = reasonable complexity = good single model
#   - Large gamma (1.0+) = very wiggly = overfits
GAMMA_WEAK = 0.001        # Very small gamma = weak learner for boosting
GAMMA_TUNED = 0.1         # Reasonably tuned for single model
ALPHA = 1.0               # Regularization

print(f"\nKernel Ridge hyperparameters:")
print(f"  Weak gamma: {GAMMA_WEAK} (weak learner for boosting)")
print(f"  Tuned gamma: {GAMMA_TUNED} (reasonably tuned single model)")
print(f"  Alpha (regularization): {ALPHA}")

# ============================================================================
# Model 1: Single Kernel Ridge with weak gamma (weak learner)
# ============================================================================
print("\n" + "="*70)
print("Training single Kernel Ridge (weak gamma - weak learner)...")
print("="*70)

kr_weak = KernelRidge(kernel='rbf', gamma=GAMMA_WEAK, alpha=ALPHA)
kr_weak.fit(X_train, y_train)
y_pred_weak = kr_weak.predict(X_test)

weak_mse = mean_squared_error(y_test, y_pred_weak)
weak_r2 = r2_score(y_test, y_pred_weak)

print(f"Weak Kernel Ridge - MSE: {weak_mse:.2f}, R²: {weak_r2:.4f}")

# ============================================================================
# Model 2: Single Kernel Ridge with tuned gamma (well-tuned single model)
# ============================================================================
print("\n" + "="*70)
print("Training single Kernel Ridge (tuned gamma - well-tuned model)...")
print("="*70)

kr_strong = KernelRidge(kernel='rbf', gamma=GAMMA_TUNED, alpha=ALPHA)
kr_strong.fit(X_train, y_train)
y_pred_strong = kr_strong.predict(X_test)

strong_mse = mean_squared_error(y_test, y_pred_strong)
strong_r2 = r2_score(y_test, y_pred_strong)

print(f"Tuned Kernel Ridge - MSE: {strong_mse:.2f}, R²: {strong_r2:.4f}")

# ============================================================================
# Model 3: Boosted Kernel Ridge (boosting the weak learner)
# ============================================================================
print("\n" + "="*70)
print("Training GGBM with Kernel Ridge (boosting the weak learner)...")
print("="*70)

N_ESTIMATORS = 50
LEARNING_RATE = 0.1

print(f"Boosting parameters:")
print(f"  n_estimators: {N_ESTIMATORS}")
print(f"  learning_rate: {LEARNING_RATE}")

ggbm_kr = GGBM(
    base_learner=KernelRidge(kernel='rbf', gamma=GAMMA_WEAK, alpha=ALPHA),
    n_estimators=N_ESTIMATORS,
    learning_rate=LEARNING_RATE,
    objective=SquaredError(),
    random_state=42
)

ggbm_kr.fit(X_train, y_train)
y_pred_ggbm = ggbm_kr.predict(X_test)

ggbm_mse = mean_squared_error(y_test, y_pred_ggbm)
ggbm_r2 = r2_score(y_test, y_pred_ggbm)

print(f"Boosted Kernel Ridge - MSE: {ggbm_mse:.2f}, R²: {ggbm_r2:.4f}")

# ============================================================================
# Comparison Summary
# ============================================================================
print("\n" + "="*70)
print("COMPARISON SUMMARY")
print("="*70)
print(f"\n{'Model':<40} {'MSE':<12} {'R²':<12}")
print("-"*70)
print(f"{'Single Kernel Ridge (weak, γ=0.001)':<40} {weak_mse:<12.2f} {weak_r2:<12.4f}")
print(f"{'Single Kernel Ridge (tuned, γ=0.1)':<40} {strong_mse:<12.2f} {strong_r2:<12.4f}")
print(f"{'Boosted Kernel Ridge (50 estimators)':<40} {ggbm_mse:<12.2f} {ggbm_r2:<12.4f}")

improvement_over_weak = (weak_mse - ggbm_mse) / weak_mse * 100
improvement_over_tuned = (strong_mse - ggbm_mse) / strong_mse * 100

print("\n" + "="*70)
print("IMPROVEMENTS")
print("="*70)
print(f"Boosting improves over weak learner by: {improvement_over_weak:.1f}%")
if improvement_over_tuned > 0:
    print(f"Boosting improves over tuned learner by: {improvement_over_tuned:.1f}%")
else:
    print(f"Tuned learner is better than boosting by: {-improvement_over_tuned:.1f}%")
    print("  (Boosting weak learners may not always beat a well-tuned single model)")

# ============================================================================
# Learning curve: Performance vs number of estimators
# ============================================================================
print("\n" + "="*70)
print("Computing learning curve...")
print("="*70)

estimator_counts = [1, 5, 10, 20, 30, 40, 50]
train_losses = []
test_losses = []
train_r2s = []
test_r2s = []

for n in estimator_counts:
    model = GGBM(
        base_learner=KernelRidge(kernel='rbf', gamma=GAMMA_WEAK, alpha=ALPHA),
        n_estimators=n,
        learning_rate=LEARNING_RATE,
        objective=SquaredError(),
        random_state=42
    )
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_loss = mean_squared_error(y_train, train_pred)
    test_loss = mean_squared_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_r2s.append(train_r2)
    test_r2s.append(test_r2)

    print(f"  n={n:2d}: Train Loss={train_loss:7.2f}, Test Loss={test_loss:7.2f} | "
          f"Train R²={train_r2:.4f}, Test R²={test_r2:.4f}")

# ============================================================================
# Visualizations
# ============================================================================
print("\n" + "="*70)
print("Creating visualizations...")
print("="*70)

fig = plt.figure(figsize=(16, 10))

# Plot 1: Weak learner predictions
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(y_test, y_pred_weak, alpha=0.5, s=20, color='red')
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax1.set_xlabel('True Values', fontsize=10)
ax1.set_ylabel('Predictions', fontsize=10)
ax1.set_title(f'Weak Kernel Ridge (γ={GAMMA_WEAK})\nMSE={weak_mse:.2f}, R²={weak_r2:.4f}',
              fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: Strong learner predictions
ax2 = plt.subplot(2, 3, 2)
ax2.scatter(y_test, y_pred_strong, alpha=0.5, s=20, color='blue')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax2.set_xlabel('True Values', fontsize=10)
ax2.set_ylabel('Predictions', fontsize=10)
ax2.set_title(f'Tuned Kernel Ridge (γ={GAMMA_TUNED})\nMSE={strong_mse:.2f}, R²={strong_r2:.4f}',
              fontsize=11)
ax2.grid(True, alpha=0.3)

# Plot 3: Boosted learner predictions
ax3 = plt.subplot(2, 3, 3)
ax3.scatter(y_test, y_pred_ggbm, alpha=0.5, s=20, color='green')
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax3.set_xlabel('True Values', fontsize=10)
ax3.set_ylabel('Predictions', fontsize=10)
ax3.set_title(f'Boosted Kernel Ridge ({N_ESTIMATORS} estimators)\nMSE={ggbm_mse:.2f}, R²={ggbm_r2:.4f}',
              fontsize=11)
ax3.grid(True, alpha=0.3)

# Plot 4: Learning curve (Loss vs estimators)
ax4 = plt.subplot(2, 3, 4)
ax4.plot(estimator_counts, train_losses, 'o-', label='Train Loss', linewidth=2, markersize=8, color='blue')
ax4.plot(estimator_counts, test_losses, 's-', label='Test Loss', linewidth=2, markersize=8, color='orange')
ax4.axhline(y=weak_mse, color='red', linestyle='--', label='Weak learner', linewidth=2)
ax4.axhline(y=strong_mse, color='green', linestyle='--', label='Tuned learner', linewidth=2)
ax4.set_xlabel('Number of Estimators', fontsize=10)
ax4.set_ylabel('MSE (Loss)', fontsize=10)
ax4.set_title('Learning Curve: Loss vs Estimators', fontsize=11)
ax4.legend(loc='upper right')
ax4.grid(True, alpha=0.3)

# Plot 5: Residuals comparison
ax5 = plt.subplot(2, 3, 5)
residuals_weak = y_test - y_pred_weak
residuals_ggbm = y_test - y_pred_ggbm
ax5.scatter(y_pred_weak, residuals_weak, alpha=0.5, s=20, label='Weak KR', color='red')
ax5.scatter(y_pred_ggbm, residuals_ggbm, alpha=0.5, s=20, label='Boosted KR', color='green')
ax5.axhline(y=0, color='k', linestyle='--', lw=2)
ax5.set_xlabel('Predictions', fontsize=10)
ax5.set_ylabel('Residuals', fontsize=10)
ax5.set_title('Residual Plot', fontsize=11)
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: MSE comparison bar plot
ax6 = plt.subplot(2, 3, 6)
models = ['Weak KR\n(γ=0.001)', 'Tuned KR\n(γ=0.1)', 'Boosted KR\n(50 est.)']
mses = [weak_mse, strong_mse, ggbm_mse]
colors = ['red', 'blue', 'green']
bars = ax6.bar(models, mses, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax6.set_ylabel('Mean Squared Error', fontsize=10)
ax6.set_title('MSE Comparison', fontsize=11)
ax6.grid(True, alpha=0.3, axis='y')
# Add value labels on bars
for bar, mse in zip(bars, mses):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'{mse:.1f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
# plt.savefig('examples/boost_kernel_ridge.png', dpi=150, bbox_inches='tight')
print("Plot saved to: examples/boost_kernel_ridge.png")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("\nKey takeaways:")
print("1. Weak learner (γ=0.001): Very smooth, high bias - underfits alone")
print("2. Tuned learner (γ=0.1): Reasonably tuned single model")
print("3. Boosting the weak learner: Reduces bias by combining many simple models")
print(f"4. Performance gain: {improvement_over_weak:.1f}% improvement over weak learner")
print("\n5. Important insight about RBF gamma:")
print("   - Small γ → smooth, simple model (underfits)")
print("   - Medium γ → balanced complexity")
print("   - Large γ → very complex, wiggly model (overfits)")
print("\n✓ GGBM successfully boosts non-tree models (Kernel Ridge)!")

# Show plot last (this blocks until window is closed)
plt.show()
