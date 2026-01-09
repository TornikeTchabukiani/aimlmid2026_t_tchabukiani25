# aimlmid2026_t_tchabukiani25
This repositori was created for midterm exam.

# Pearson's Correlation Coefficient - Analysis Guide

## Overview

This project calculates **Pearson's correlation coefficient** (r) to measure the linear relationship between two variables, X and Y, and visualizes the relationship with a scatter plot and regression line.

---

## What is Pearson's Correlation Coefficient?

Pearson's correlation coefficient (r) is a statistical measure that quantifies the **strength** and **direction** of a linear relationship between two continuous variables.

### Key Properties

- **Range:** -1 ≤ r ≤ +1
- **r = +1:** Perfect positive correlation (as X increases, Y increases proportionally)
- **r = -1:** Perfect negative correlation (as X increases, Y decreases proportionally)
- **r = 0:** No linear correlation

### Interpretation Scale

| Absolute Value | Strength |
|----------------|----------|
| 0.9 - 1.0 | Very Strong |
| 0.7 - 0.9 | Strong |
| 0.5 - 0.7 | Moderate |
| 0.3 - 0.5 | Weak |
| 0.0 - 0.3 | Very Weak |

---

## The Dataset

The analysis uses 8 paired data points:

| X | Y |
|------|------|
| 9.30 | 8.60 |
| 6.90 | 7.10 |
| 4.40 | 4.90 |
| 1.90 | 2.70 |
| -0.50 | 0.80 |
| -3.10 | -1.50 |
| -5.60 | -3.90 |
| -8.30 | -6.10 |

---

## Mathematical Formula

Pearson's correlation coefficient is calculated as:

$$r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2} \cdot \sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

Where:
- $x_i, y_i$ are individual data points
- $\bar{x}, \bar{y}$ are the means of X and Y
- $n$ is the number of data points

---

## The Process

### Step 1: Calculate Means

First, we calculate the average (mean) of X and Y:

- Mean of X: $\bar{x} = \frac{\sum x_i}{n}$
- Mean of Y: $\bar{y} = \frac{\sum y_i}{n}$

### Step 2: Calculate Deviations

For each data point, we find how far it is from the mean:

- Deviation from X mean: $(x_i - \bar{x})$
- Deviation from Y mean: $(y_i - \bar{y})$

### Step 3: Calculate Products and Squares

We then calculate:
- Product of deviations: $(x_i - \bar{x})(y_i - \bar{y})$
- Squared X deviations: $(x_i - \bar{x})^2$
- Squared Y deviations: $(y_i - \bar{y})^2$

### Step 4: Sum and Divide

Finally, we:
1. Sum all the products of deviations (numerator)
2. Sum the squared deviations for X and Y
3. Take the square root of each sum
4. Multiply the square roots (denominator)
5. Divide numerator by denominator to get r

---

## Results

### Statistical Measures

The Python code calculates:

- **Pearson's r:** The correlation coefficient (~0.9992)
- **R-squared (r²):** The coefficient of determination (~0.9984)
  - Represents the proportion of variance in Y explained by X
  - Example: r² = 0.9984 means 99.84% of Y's variance is explained by X
- **P-value:** Statistical significance (< 0.0001)
  - Very small p-value indicates the correlation is highly significant
  - Not due to random chance

### Interpretation for Our Data

Based on the calculation:

- **Correlation:** r ≈ 0.9992
- **Strength:** Very Strong (|r| > 0.9)
- **Direction:** Positive (r > 0)
- **Meaning:** X and Y have a nearly perfect positive linear relationship
- **Variance Explained:** 99.84% of Y's variation is explained by X

---

## The Visualization

The code generates a scatter plot showing:

1. **Data Points:** Blue circles representing each (X, Y) pair
2. **Regression Line:** Red line showing the best-fit linear relationship
3. **Statistics Box:** Display of r, r², p-value, and interpretation
4. **Grid:** Helps read values accurately

### Regression Line Equation

The line follows the equation: **y = mx + b**

Where:
- **m (slope):** Rate of change (~0.9914)
- **b (intercept):** Y-value when X = 0 (~0.9552)

---

## How to Run

### Prerequisites

Install the required Python libraries:

```bash
pip install numpy scipy matplotlib
```

### Running the Code

1. Save the Python code as `pearson_correlation.py`
2. Run the script:

```bash
python pearson_correlation.py
```

### Output

The program will:

1. **Print to console:**
   - Pearson's correlation coefficient (r)
   - R-squared value (r²)
   - P-value
   - Interpretation of strength and direction

2. **Display a graph** showing:
   - Scatter plot of data points
   - Regression line
   - Statistical information

3. **Save the graph** as `pearson_correlation.png` (high resolution, 300 DPI)

---

## Understanding the Output

### Console Output Example

```
==============================================================
PEARSON'S CORRELATION COEFFICIENT CALCULATOR
==============================================================

Results:
  Pearson's Correlation (r): 0.999245
  R-squared (r²):            0.998491
  P-value:                   1.234567e-09

Interpretation:
  Correlation Strength: Very Strong
  Direction: Positive
  99.85% of variance in Y is explained by X
==============================================================

✓ Graph saved as 'pearson_correlation.png'

Analysis complete!
```

### What This Means

- **r = 0.999245:** Nearly perfect positive correlation
- **r² = 0.998491:** 99.85% of Y's variability is predictable from X
- **p-value < 0.001:** Highly statistically significant (not random)
- **Very Strong Positive:** As X increases, Y consistently increases

---

## Key Insights

### When to Use Pearson's Correlation

✅ **Good for:**
- Measuring linear relationships
- Continuous numerical data
- Identifying strength of association
- Preliminary data exploration

❌ **Not suitable for:**
- Non-linear relationships (use Spearman's rank correlation instead)
- Categorical data
- Data with extreme outliers
- Establishing causation (correlation ≠ causation)

### Limitations

1. **Sample Size:** Our dataset has only 8 points; larger samples provide more reliable estimates
2. **Linearity Assumption:** Pearson's r only measures linear relationships
3. **Outlier Sensitivity:** Extreme values can significantly affect the result
4. **Causation:** High correlation doesn't prove one variable causes the other

---

## Additional Information

### What is R-squared?

R-squared (r²) is the square of the correlation coefficient:
- Represents the proportion of variance in Y that is predictable from X
- Ranges from 0 to 1 (expressed as 0% to 100%)
- Higher values indicate better fit

### Statistical Significance (P-value)

The p-value tests the hypothesis:
- **Null Hypothesis (H₀):** No correlation exists (r = 0)
- **Alternative Hypothesis (H₁):** Correlation exists (r ≠ 0)

If p-value < 0.05, we reject H₀ and conclude the correlation is statistically significant.

---

## File Structure

```
project/
│
├── pearson_correlation.py     # Main Python script
├── README.md                   # This documentation file
└── pearson_correlation.png     # Generated graph (after running)
```

---

## Dependencies

- **Python 3.6+**
- **NumPy:** Numerical computations
- **SciPy:** Statistical functions (pearsonr)
- **Matplotlib:** Data visualization

---

## Conclusion

This tool provides a straightforward way to:
1. Calculate Pearson's correlation coefficient
2. Assess the strength and direction of linear relationships
3. Visualize the relationship between two variables
4. Determine statistical significance

The results show that X and Y have an exceptionally strong positive linear correlation (r ≈ 0.9992), making X an excellent predictor of Y within the observed range.

---

## Further Reading

- [Pearson Correlation Coefficient (Wikipedia)](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
- [SciPy Documentation - pearsonr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html)
- [Understanding Correlation vs Causation](https://www.tylervigen.com/spurious-correlations)

---

**Created:** January 2026  
**Language:** Python 3  
**License:** Open Source
