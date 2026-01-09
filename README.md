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

<img width="1000" height="600" alt="Linear regression Graph" src="https://github.com/user-attachments/assets/1ea64f6d-2844-4abe-ad5f-d1263be4ff15" />


