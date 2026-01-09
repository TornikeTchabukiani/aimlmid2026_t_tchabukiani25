import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Dataset
x = np.array([9.30, 6.90, 4.40, 1.90, -0.50, -3.10, -5.60, -8.30])
y = np.array([8.60, 7.10, 4.90, 2.70, 0.80, -1.50, -3.90, -6.10])

print("=" * 60)
print("PEARSON'S CORRELATION COEFFICIENT CALCULATOR")
print("=" * 60)

# Calculate Pearson's correlation coefficient
correlation, p_value = pearsonr(x, y)
r_squared = correlation ** 2

# Display results
print(f"\nResults:")
print(f"  Pearson's Correlation (r): {correlation:.6f}")
print(f"  R-squared (r²):            {r_squared:.6f}")
print(f"  P-value:                   {p_value:.6e}")

# Interpretation
print(f"\nInterpretation:")
if abs(correlation) >= 0.9:
    strength = "Very Strong"
elif abs(correlation) >= 0.7:
    strength = "Strong"
elif abs(correlation) >= 0.5:
    strength = "Moderate"
elif abs(correlation) >= 0.3:
    strength = "Weak"
else:
    strength = "Very Weak"

direction = "Positive" if correlation > 0 else "Negative"
print(f"  Correlation Strength: {strength}")
print(f"  Direction: {direction}")
print(f"  {r_squared*100:.2f}% of variance in Y is explained by X")

print("=" * 60)

# Calculate regression line
slope = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x))**2)
intercept = np.mean(y) - slope * np.mean(x)

# Create the plot
plt.figure(figsize=(10, 6))

# Scatter plot
plt.scatter(x, y, color='#4F46E5', s=100, alpha=0.7,
            edgecolors='black', linewidth=2, label='Data Points', zorder=3)

# Regression line
x_line = np.linspace(x.min(), x.max(), 100)
y_line = slope * x_line + intercept
plt.plot(x_line, y_line, color='#EF4444', linewidth=2.5,
         label=f'Regression Line: y = {slope:.4f}x + {intercept:.4f}', zorder=2)

# Formatting
plt.xlabel('X Variable', fontsize=12, fontweight='bold')
plt.ylabel('Y Variable', fontsize=12, fontweight='bold')
plt.title('Pearson\'s Correlation Analysis', fontsize=14, fontweight='bold', pad=15)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='best', fontsize=10)

# Add statistics box
textstr = f'r = {correlation:.4f}\nr² = {r_squared:.4f}\np-value = {p_value:.2e}\n\n{strength} {direction}\nCorrelation'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=1.5)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top', bbox=props, family='monospace')

plt.tight_layout()
plt.savefig('pearson_correlation.png', dpi=300, bbox_inches='tight')
print("\n✓ Graph saved as 'pearson_correlation.png'")
plt.show()

print("\nAnalysis complete!")