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












# Email Spam Classification System

A machine learning project that uses **Logistic Regression** to classify emails as spam or legitimate based on extracted features.

---

## 📊 Project Overview

- **Model**: Logistic Regression
- **Dataset Size**: 2,500 emails
- **Accuracy**: 96.53%
- **Programming Language**: Python
- **Libraries**: scikit-learn, pandas, numpy, matplotlib, seaborn

---

## 📁 Table of Contents

1. [Data Upload and Repository Link](#1-data-upload-and-repository-link)
2. [Model Training](#2-model-training)
3. [Model Validation](#3-model-validation)
4. [Email Text Classification](#4-email-text-classification)
5. [Spam Email Example](#5-spam-email-example)
6. [Legitimate Email Example](#6-legitimate-email-example)
7. [Data Visualizations](#7-data-visualizations)
8. [Conclusion](#8-conclusion)
9. [How to Run](#how-to-run)

---

## 1. Data Upload and Repository Link

### Dataset Information

The email classification dataset contains **2,500 email samples** with the following features:

| Feature | Description |
|---------|-------------|
| **words** | Number of words in the email |
| **links** | Number of hyperlinks |
| **capital_words** | Number of words in ALL CAPS |
| **spam_word_count** | Count of spam-related keywords |
| **is_spam** | Target variable (0 = legitimate, 1 = spam) |

### Dataset Location
```
/t_tchabukiani25_16928.csv
```

### Dataset Statistics

| Metric | Words | Links | Capital Words | Spam Words |
|--------|-------|-------|---------------|------------|
| **Mean** | 323.88 | 3.21 | 8.40 | 3.60 |
| **Std Dev** | 255.36 | 2.91 | 8.94 | 3.08 |
| **Min** | 20 | 0 | 0 | 0 |
| **Max** | 1000 | 10 | 30 | 10 |

### Class Distribution

- **Legitimate emails**: 1,257 (50.28%)
- **Spam emails**: 1,243 (49.72%)

> ✅ The dataset is **well-balanced**, which is ideal for training a binary classification model without bias.

---

## 2. Model Training

### 2.1 Data Loading and Processing Code

The dataset is split into **70% training** and **30% testing** sets:
```python
def load_and_process_data(self):
    # Load the dataset
    df = pd.read_csv(self.data_path)
    
    # Separate features and target
    X = df.drop('is_spam', axis=1)
    y = df['is_spam']
    
    # Split data: 70% training, 30% testing
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    return self.X_train, self.X_test, self.y_train, self.y_test
```

**Source Code**: `spamEmailDetection.py` (lines 41-68)

**Result**:
- Training set: **1,750 samples (70%)**
- Testing set: **750 samples (30%)**

### 2.2 Model Training Code

The logistic regression model is trained using scikit-learn:
```python
def train_model(self):
    # Create and train the logistic regression model
    self.model = LogisticRegression(
        random_state=42, 
        max_iter=1000, 
        solver='lbfgs'
    )
    self.model.fit(self.X_train, self.y_train)
    
    # Extract coefficients
    self.coefficients = self.model.coef_[0]
    intercept = self.model.intercept_[0]
    
    return self.model
```

**Source Code**: `spamEmailDetection.py` (lines 70-93)

### 2.3 Model Coefficients

The trained model produced the following coefficients:

| Parameter | Coefficient Value |
|-----------|-------------------|
| **Intercept** | -9.533241 |
| **words** | 0.007841 |
| **links** | 0.909028 |
| **capital_words** | 0.459657 |
| **spam_word_count** | 0.766031 |

### 2.4 Coefficient Interpretation

#### Analysis of Feature Importance:

1. **links (0.909028)** ⭐ **Strongest Predictor**
   - Emails with more hyperlinks are significantly more likely to be spam
   - This is the most influential feature in the model

2. **spam_word_count (0.766031)** ⭐ **Second Strongest**
   - Higher counts of spam-related keywords strongly indicate spam
   - Keywords include: "free", "winner", "cash", "urgent", "guaranteed"

3. **capital_words (0.459657)** 🔸 **Moderate Effect**
   - Excessive capitalization is a common spam tactic (e.g., "BUY NOW!")
   - Moderate positive correlation with spam

4. **words (0.007841)** 🔹 **Weakest Predictor**
   - Email length has minimal impact on classification
   - Spam can be short or long

5. **Intercept (-9.533241)** 🔒 **Baseline**
   - Large negative value ensures the model defaults to "legitimate"
   - Strong spam indicators are required to override this baseline

---

## 3. Model Validation

### 3.1 Validation Code

The model was validated on the **30% test set** (750 emails):
```python
def validate_model(self):
    # Make predictions on test set
    y_pred = self.model.predict(self.X_test)
    
    # Calculate confusion matrix
    cm = confusion_matrix(self.y_test, y_pred)
    
    # Calculate accuracy
    accuracy = accuracy_score(self.y_test, y_pred)
    
    return cm, accuracy
```

**Source Code**: `spamEmailDetection.py` (lines 95-133)

### 3.2 Confusion Matrix

|  | **Predicted Legitimate** | **Predicted Spam** |
|---|---|---|
| **Actual Legitimate** | 366 ✅ | 11 ❌ |
| **Actual Spam** | 15 ❌ | 358 ✅ |

#### Confusion Matrix Breakdown:

- **True Negatives (TN) = 366**: Legitimate emails correctly classified as legitimate
- **False Positives (FP) = 11**: Legitimate emails incorrectly classified as spam (Type I error)
- **False Negatives (FN) = 15**: Spam emails incorrectly classified as legitimate (Type II error)
- **True Positives (TP) = 358**: Spam emails correctly classified as spam

### 3.3 Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Accuracy** | **96.53%** | Overall correctness of the model |
| **Precision** | 97.02% | Of emails marked as spam, 97% are actually spam |
| **Recall** | 95.98% | Of all spam emails, 96% are caught |
| **F1-Score** | 96.50% | Harmonic mean of precision and recall |

### Performance Analysis

✅ **Excellent Performance**: The model achieves **96.53% accuracy** on unseen data.

✅ **High Precision (97.02%)**: Very few false positives means legitimate emails are rarely marked as spam. This is critical for user experience.

✅ **Strong Recall (95.98%)**: The model catches most spam emails, with only 15 spam emails slipping through in the test set.

✅ **Balanced Errors**: The low error rates in both directions demonstrate the model works well for both spam detection and legitimate email preservation.

---

## 4. Email Text Classification

The application can classify raw email text by extracting features automatically.

### 4.1 Feature Extraction Process
```python
def extract_features_from_text(self, email_text):
    spam_keywords = [
        'free', 'winner', 'cash', 'prize', 'click', 'buy',
        'limited', 'urgent', 'guaranteed', 'bonus', 'discount',
        'congratulations', 'act now', 'call now', 'unsubscribe'
    ]
    
    # Count total words
    words = len(email_text.split())
    
    # Count links (URLs)
    link_pattern = r'http[s]?://...'
    links = len(re.findall(link_pattern, email_text))
    
    # Count capitalized words (ALL CAPS with 2+ letters)
    capital_words = len(re.findall(r'\b[A-Z]{2,}\b', email_text))
    
    # Count spam words
    email_lower = email_text.lower()
    spam_word_count = sum(1 for kw in spam_keywords 
                         if kw in email_lower)
    
    return {'words': words, 'links': links, 
            'capital_words': capital_words, 
            'spam_word_count': spam_word_count}
```

**Source Code**: `spamEmailDetection.py` (lines 135-171)

### 4.2 Feature Extraction Components

| Component | Method |
|-----------|--------|
| **Word Count** | Split text by whitespace |
| **Link Detection** | Regex pattern matching for HTTP/HTTPS URLs |
| **Capital Words** | Regex to find words with 2+ consecutive uppercase letters |
| **Spam Keywords** | Match against dictionary of 24 common spam words |

---

## 5. Spam Email Example

### 5.1 Composed Spam Email
```
CONGRATULATIONS!!! You are a WINNER!

Click here NOW to claim your FREE $1,000,000 CASH PRIZE!

This is a LIMITED TIME OFFER - ACT NOW!
Visit: http://free-money-winner.com http://claim-prize-now.com

You have been selected to receive a GUARANTEED BONUS of $50,000!
BUY NOW and SAVE BIG! Special discount available!

URGENT: This offer expires TODAY! Don't miss this amazing deal!
Click http://urgent-offer.com to get your FREE credit card with no fees!

Call NOW: 1-800-WINNER
Unsubscribe: http://unsubscribe.com
```

### 5.2 Classification Result

| Feature | Extracted Value |
|---------|----------------|
| **words** | 72 |
| **links** | 4 |
| **capital_words** | 22 |
| **spam_word_count** | 21 |
| **Classification** | **SPAM (100.00% confidence)** |

### 5.3 Explanation: How This Email Was Designed as Spam

This email was deliberately crafted with classic spam characteristics:

#### 1. 🔗 **Excessive Links (4 URLs)**
- Multiple suspicious URLs trigger the highest-weighted feature (coefficient: 0.909)
- Spam emails often contain many links to phishing sites or promotional pages
- URLs like "free-money-winner.com" and "claim-prize-now.com" are typical spam domains

#### 2. 🎯 **High Spam Keyword Count (21 keywords)**
- Contains: 'FREE', 'WINNER', 'CASH', 'PRIZE', 'URGENT', 'GUARANTEED', 'BONUS', 'LIMITED', 'BUY NOW', 'DISCOUNT', 'CALL NOW', 'UNSUBSCRIBE'
- Triggers the second-strongest predictor (coefficient: 0.766)
- These words are classic spam indicators

#### 3. 📢 **Extensive Capitalization (22 words in ALL CAPS)**
- Words like: CONGRATULATIONS, WINNER, NOW, FREE, CASH, PRIZE, LIMITED, ACT, GUARANTEED, BUY, SAVE, URGENT, TODAY
- All-caps text is used to grab attention—a common spam tactic
- Coefficient: 0.460

#### 4. ⚠️ **Urgency and Pressure Tactics**
- "ACT NOW", "LIMITED TIME OFFER", "expires TODAY"
- Creates artificial urgency—a hallmark of spam

#### 5. 💰 **Unrealistic Promises**
- "$1,000,000 CASH PRIZE" and "$50,000 BONUS"
- Typical of fraudulent spam emails

**Result**: The model classified this email as **SPAM with 100.00% confidence** ✅

---

## 6. Legitimate Email Example

### 6.1 Composed Legitimate Email
```
Dear Team,

I hope this message finds you well. I wanted to follow up on our meeting
yesterday regarding the project timeline.

As discussed, we need to finalize the requirements document by next Friday.
Please review the attached draft and send me your feedback by Wednesday.

The key points we agreed on:
- Complete initial testing phase by end of month
- Schedule review meeting with stakeholders
- Update documentation for the new features

Let me know if you have any questions or concerns.

Best regards,
John Smith
Project Manager
```

### 6.2 Classification Result

| Feature | Extracted Value |
|---------|----------------|
| **words** | 90 |
| **links** | 0 |
| **capital_words** | 0 |
| **spam_word_count** | 0 |
| **Classification** | **LEGITIMATE (99.99% confidence)** |

### 6.3 Explanation: How This Email Was Designed as Legitimate

This email exhibits characteristics of professional business communication:

#### 1. 🚫 **No Links (0 URLs)**
- Avoids the strongest spam indicator
- Legitimate internal business emails rarely include external URLs
- Coefficient impact: 0 × 0.909 = 0

#### 2. 📝 **No Spam Keywords (0 spam words)**
- Uses professional business language instead
- Words used: "meeting", "project", "timeline", "requirements", "feedback", "stakeholders", "documentation"
- No urgency tactics, no promotional language

#### 3. ✏️ **Proper Capitalization (0 ALL CAPS)**
- Uses correct grammar and capitalization throughout
- Only appropriate proper nouns are capitalized
- No attention-grabbing ALL CAPS style

#### 4. 🎓 **Professional Tone**
- Courteous greeting: "Dear Team"
- Organized content with clear bullet points
- Professional closing: "Best regards"
- Includes name and title

#### 5. 🎯 **Specific Context**
- References specific events: "meeting yesterday"
- Concrete deliverables: "requirements document"
- Reasonable deadlines: "next Friday", "Wednesday"

#### 6. 🤝 **No Pressure Tactics**
- No artificial urgency
- No unrealistic promises
- No pressure to act immediately
- Polite request: "Let me know if you have any questions"

**Result**: The model classified this email as **LEGITIMATE with 99.99% confidence** ✅

---

## 7. Data Visualizations

All visualizations were generated using **matplotlib** and **seaborn** libraries.

### 7.1 Visualization 1: Class Distribution Analysis

#### Python Code:
```python
def visualize_class_distribution(self):
    df = pd.read_csv(self.data_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    class_counts = df['is_spam'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    bars = ax1.bar(['Legitimate', 'Spam'], 
                   [class_counts[0], class_counts[1]], 
                   color=colors, edgecolor='black', linewidth=1.5)
    
    ax1.set_xlabel('Email Class', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Emails', fontsize=12, fontweight='bold')
    ax1.set_title('Class Distribution: Spam vs. Legitimate Emails',
                 fontsize=14, fontweight='bold', pad=20)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Pie chart
    ax2.pie([class_counts[0], class_counts[1]], 
            labels=['Legitimate', 'Spam'],
            autopct='%1.1f%%', colors=colors,
            startangle=90, explode=(0.05, 0.05))
    
    ax2.set_title('Class Distribution Percentage',
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('outputs/visualization_class_distribution.png', dpi=300)
```

**Source Code**: `spamEmailDetection.py` (lines 213-255)

#### Generated Visualization:

![Class Distribution](outputs/visualization_class_distribution.png)

#### Insights:

The visualization reveals that the dataset is **remarkably well-balanced**, with 1,257 legitimate emails (50.28%) and 1,243 spam emails (49.72%). This near-perfect balance is highly beneficial for model training, as it prevents bias toward either class and ensures the model learns equally from both spam and legitimate email patterns. Balanced datasets typically yield more reliable classification performance across both classes, which is reflected in our high accuracy metrics.

---

### 7.2 Visualization 2: Confusion Matrix Heatmap

#### Python Code:
```python
def visualize_confusion_matrix(self, cm):
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Spam'],
                yticklabels=['Legitimate', 'Spam'],
                cbar_kws={'label': 'Count'},
                linewidths=2, linecolor='black',
                annot_kws={'size': 16, 'fontweight': 'bold'})
    
    plt.xlabel('Predicted Class', fontsize=13, fontweight='bold')
    plt.ylabel('Actual Class', fontsize=13, fontweight='bold')
    plt.title('Confusion Matrix Heatmap\nEmail Spam Classification',
             fontsize=15, fontweight='bold', pad=20)
    
    # Add accuracy text
    accuracy = (cm[0][0] + cm[1][1]) / cm.sum()
    plt.text(0.5, -0.15, f'Overall Accuracy: {accuracy:.2%}',
            transform=plt.gca().transAxes,
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', 
                     facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('outputs/visualization_confusion_matrix.png', dpi=300)
```

**Source Code**: `spamEmailDetection.py` (lines 257-287)

#### Generated Visualization:

![Confusion Matrix](outputs/visualization_confusion_matrix.png)

#### Insights:

The confusion matrix heatmap reveals **excellent model performance** with strong diagonal values (366 and 358), indicating correct classifications. The model demonstrates only **11 false positives** (legitimate emails incorrectly flagged as spam) and **15 false negatives** (spam emails that slipped through). This asymmetry is actually desirable in spam detection—false positives are more problematic than false negatives, as users would rather receive occasional spam than have important legitimate emails blocked. Our model's low false positive rate (2.9%) makes it practical for real-world deployment.

---

### 7.3 Visualization 3: Feature Importance and Correlation

#### Python Code:
```python
def visualize_feature_importance(self):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Feature Coefficients Bar Chart
    colors = ['#3498db' if coef < 0 else '#e74c3c' 
              for coef in self.coefficients]
    bars = ax1.barh(self.feature_names, self.coefficients,
                    color=colors, edgecolor='black', linewidth=1.5)
    
    ax1.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Features', fontsize=12, fontweight='bold')
    ax1.set_title('Logistic Regression Feature Coefficients',
                 fontsize=14, fontweight='bold', pad=20)
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
    
    # Feature Correlation Heatmap
    df = pd.read_csv(self.data_path)
    correlation_matrix = df.corr()
    
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f',
               cmap='coolwarm', center=0, square=True,
               linewidths=1, linecolor='black',
               cbar_kws={'label': 'Correlation'},
               ax=ax2, vmin=-1, vmax=1)
    
    ax2.set_title('Feature Correlation Heatmap',
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('outputs/visualization_feature_importance.png', dpi=300)
```

**Source Code**: `spamEmailDetection.py` (lines 289-344)

#### Generated Visualization:

<img width="3890" height="1468" alt="visualization_class_distribution" src="https://github.com/user-attachments/assets/3204de7a-bacf-4ca2-bd5c-df83ba218a71" />

<img width="2813" height="2358" alt="visualization_confusion_matrix" src="https://github.com/user-attachments/assets/7555c0c3-4c4e-4e78-8a93-8ca3691e14b7" />

<img width="4737" height="1787" alt="visualization_feature_importance" src="https://github.com/user-attachments/assets/f96ff934-f2a6-4320-9702-1a16e66a6a05" />



#### Insights:

The feature importance visualization shows that **'links'** has the highest coefficient (0.909), making it the most influential predictor of spam, followed by **'spam_word_count'** (0.766). The correlation heatmap reveals that features are relatively independent, with the strongest correlation being between 'spam_word_count' and 'is_spam' (0.69), confirming that spam-related keywords are indeed highly predictive of spam emails. The low inter-feature correlations (most below 0.40) suggest minimal multicollinearity, which is ideal for logistic regression and indicates that each feature contributes unique information to the model.

---

## 8. Conclusion

### Project Achievements

This project successfully developed a **high-performance email spam classification system** using logistic regression:

✅ **Model Accuracy**: Achieved **96.53% accuracy** on unseen test data

✅ **Feature Engineering**: Successfully implemented automated feature extraction from raw email text

✅ **Practical Application**: Created a functional console application capable of classifying real emails

✅ **Model Interpretability**: Identified that the **number of links** is the strongest spam indicator

✅ **Balanced Performance**: High precision (97.02%) and recall (95.98%) demonstrate the model works well for both spam detection and legitimate email preservation

### Key Insights

1. **Links are the strongest spam indicator** (coefficient: 0.909)
   - Spammers use multiple links to redirect users to malicious sites
   - Legitimate business emails rarely contain multiple external links

2. **Spam keywords are highly predictive** (coefficient: 0.766)
   - Words like "free", "winner", "urgent", "guaranteed" are strong indicators
   - Building a comprehensive keyword dictionary improves detection

3. **Capitalization matters** (coefficient: 0.460)
   - Excessive use of ALL CAPS is a spam tactic
   - Professional emails use proper grammar and capitalization

4. **Email length is less important** (coefficient: 0.008)
   - Both spam and legitimate emails can be short or long
   - Content quality matters more than quantity

### Model Strengths

- ✅ High accuracy with minimal false positives
- ✅ Interpretable coefficients that explain predictions
- ✅ Fast training and prediction times
- ✅ Works well with balanced datasets
- ✅ Simple to implement and maintain

### Potential Improvements

Future enhancements could include:

1. **Expand spam keyword dictionary** with more terms and phrases
2. **Add new features** such as:
   - Email metadata (sender domain, time of day)
   - Sender reputation scores
   - Email formatting patterns
   - Attachment analysis
3. **Explore ensemble methods** (Random Forest, Gradient Boosting) to potentially improve accuracy
4. **Implement real-time classification** with a web interface
5. **Add multi-language support** for international emails

### Real-World Applications

This spam classifier can be used for:

- 📧 **Email filtering systems**
- 🛡️ **Security applications**
- 📊 **Data cleaning pipelines**
- 🎓 **Educational purposes**
- 🔬 **Research projects**

---

## 👨‍💻 Author

**Student Project** - Tornike Tchabukiani

---

## 🙏 Acknowledgments

- Dataset: Email spam features dataset
- Libraries: scikit-learn, pandas, numpy, matplotlib, seaborn
- Model: Logistic Regression 

---



