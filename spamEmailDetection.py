"""
Email Spam Classification System using Logistic Regression
Author: Student
Description: Console application for classifying emails as spam or legitimate
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import re
import os
import warnings
warnings.filterwarnings('ignore')


class EmailSpamClassifier:
    """
    Email Spam Classifier using Logistic Regression
    """

    def __init__(self, data_path):
        """
        Initialize the classifier with data path

        Args:
            data_path (str): Path to the CSV data file
        """
        self.data_path = data_path
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.coefficients = None

    def load_and_process_data(self):
        """
        Load the dataset and split into training and testing sets

        Returns:
            tuple: Training and testing data splits
        """
        print("=" * 60)
        print("STEP 1: LOADING AND PROCESSING DATA")
        print("=" * 60)

        # Load the dataset
        df = pd.read_csv(self.data_path)
        print(f"\nDataset loaded successfully!")
        print(f"Total samples: {len(df)}")
        print(f"\nDataset preview:")
        print(df.head(10))

        print(f"\nDataset statistics:")
        print(df.describe())

        # Class distribution
        print(f"\nClass distribution:")
        print(df['is_spam'].value_counts())
        print(f"Spam emails: {df['is_spam'].sum()} ({df['is_spam'].sum()/len(df)*100:.2f}%)")
        print(f"Legitimate emails: {len(df) - df['is_spam'].sum()} ({(len(df) - df['is_spam'].sum())/len(df)*100:.2f}%)")

        # Separate features and target
        X = df.drop('is_spam', axis=1)
        y = df['is_spam']
        self.feature_names = X.columns.tolist()

        # Split data: 70% training, 30% testing
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )

        print(f"\nData split completed:")
        print(f"Training set size: {len(self.X_train)} samples ({len(self.X_train)/len(df)*100:.1f}%)")
        print(f"Testing set size: {len(self.X_test)} samples ({len(self.X_test)/len(df)*100:.1f}%)")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_model(self):
        """
        Train the logistic regression model

        Returns:
            LogisticRegression: Trained model
        """
        print("\n" + "=" * 60)
        print("STEP 2: TRAINING LOGISTIC REGRESSION MODEL")
        print("=" * 60)

        # Create and train the logistic regression model
        self.model = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
        self.model.fit(self.X_train, self.y_train)

        print("\nModel trained successfully!")

        # Extract coefficients
        self.coefficients = self.model.coef_[0]
        intercept = self.model.intercept_[0]

        print(f"\nModel Coefficients:")
        print(f"Intercept: {intercept:.6f}")
        print("\nFeature Coefficients:")
        for feature, coef in zip(self.feature_names, self.coefficients):
            print(f"  {feature:20s}: {coef:10.6f}")

        return self.model

    def validate_model(self):
        """
        Validate the model on test data and calculate metrics

        Returns:
            tuple: Confusion matrix and accuracy
        """
        print("\n" + "=" * 60)
        print("STEP 3: MODEL VALIDATION")
        print("=" * 60)

        # Make predictions on test set
        y_pred = self.model.predict(self.X_test)

        # Calculate confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)

        # Calculate accuracy
        accuracy = accuracy_score(self.y_test, y_pred)

        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"                 Legit  Spam")
        print(f"Actual  Legit  [{cm[0][0]:5d} {cm[0][1]:5d}]")
        print(f"        Spam   [{cm[1][0]:5d} {cm[1][1]:5d}]")

        print(f"\nDetailed Metrics:")
        print(f"True Negatives (TN):  {cm[0][0]} - Correctly classified as legitimate")
        print(f"False Positives (FP): {cm[0][1]} - Legitimate emails classified as spam")
        print(f"False Negatives (FN): {cm[1][0]} - Spam emails classified as legitimate")
        print(f"True Positives (TP):  {cm[1][1]} - Correctly classified as spam")

        print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        # Calculate additional metrics
        precision = cm[1][1] / (cm[1][1] + cm[0][1]) if (cm[1][1] + cm[0][1]) > 0 else 0
        recall = cm[1][1] / (cm[1][1] + cm[1][0]) if (cm[1][1] + cm[1][0]) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        return cm, accuracy

    def extract_features_from_text(self, email_text):
        """
        Extract features from raw email text

        Args:
            email_text (str): Raw email text

        Returns:
            dict: Extracted features
        """
        # Common spam keywords
        spam_keywords = [
            'free', 'winner', 'cash', 'prize', 'click', 'buy', 'offer',
            'limited', 'urgent', 'guaranteed', 'bonus', 'discount',
            'credit', 'money', 'earn', 'save', 'deal', 'order',
            'congratulations', 'act now', 'call now', 'subscribe',
            'unsubscribe', 'viagra', 'lottery', 'million'
        ]

        # Count total words
        words = len(email_text.split())

        # Count links (URLs)
        link_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        links = len(re.findall(link_pattern, email_text))

        # Count capitalized words (words in ALL CAPS with 2+ letters)
        capital_words = len(re.findall(r'\b[A-Z]{2,}\b', email_text))

        # Count spam words
        email_lower = email_text.lower()
        spam_word_count = sum(1 for keyword in spam_keywords if keyword in email_lower)

        features = {
            'words': words,
            'links': links,
            'capital_words': capital_words,
            'spam_word_count': spam_word_count
        }

        return features

    def classify_email(self, email_text, verbose=True):
        """
        Classify an email text as spam or legitimate

        Args:
            email_text (str): Raw email text
            verbose (bool): Whether to print detailed information

        Returns:
            tuple: (prediction, probability, features)
        """
        # Extract features
        features = self.extract_features_from_text(email_text)

        # Create feature vector in correct order
        feature_vector = np.array([[
            features['words'],
            features['links'],
            features['capital_words'],
            features['spam_word_count']
        ]])

        # Make prediction
        prediction = self.model.predict(feature_vector)[0]
        probability = self.model.predict_proba(feature_vector)[0]

        if verbose:
            print("\n" + "=" * 60)
            print("EMAIL CLASSIFICATION RESULT")
            print("=" * 60)
            print(f"\nExtracted Features:")
            for feature, value in features.items():
                print(f"  {feature:20s}: {value}")

            print(f"\nPrediction Probabilities:")
            print(f"  Legitimate: {probability[0]:.4f} ({probability[0]*100:.2f}%)")
            print(f"  Spam:       {probability[1]:.4f} ({probability[1]*100:.2f}%)")

            print(f"\nFinal Classification: {'SPAM' if prediction == 1 else 'LEGITIMATE'}")

        return prediction, probability, features

    def visualize_class_distribution(self):
        """
        Visualization A: Class Distribution Bar Chart
        """
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATION: Class Distribution")
        print("=" * 60)

        # Create outputs directory if it doesn't exist
        os.makedirs('outputs', exist_ok=True)

        # Load full dataset
        df = pd.read_csv(self.data_path)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart
        class_counts = df['is_spam'].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        bars = ax1.bar(['Legitimate', 'Spam'],
                       [class_counts[0], class_counts[1]],
                       color=colors,
                       edgecolor='black',
                       linewidth=1.5)

        ax1.set_xlabel('Email Class', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Emails', fontsize=12, fontweight='bold')
        ax1.set_title('Class Distribution: Spam vs. Legitimate Emails',
                     fontsize=14, fontweight='bold', pad=20)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}\n({height/len(df)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Pie chart
        colors_pie = ['#2ecc71', '#e74c3c']
        wedges, texts, autotexts = ax2.pie([class_counts[0], class_counts[1]],
                                            labels=['Legitimate', 'Spam'],
                                            autopct='%1.1f%%',
                                            colors=colors_pie,
                                            startangle=90,
                                            explode=(0.05, 0.05),
                                            textprops={'fontsize': 11, 'fontweight': 'bold'})

        ax2.set_title('Class Distribution Percentage',
                     fontsize=14, fontweight='bold', pad=20)

        # Add legend
        ax2.legend(wedges, [f'Legitimate ({class_counts[0]})', f'Spam ({class_counts[1]})'],
                  title="Email Class",
                  loc="best",
                  fontsize=10)

        plt.tight_layout()
        plt.savefig('outputs/visualization_class_distribution.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved: outputs/visualization_class_distribution.png")
        plt.close()

    def visualize_confusion_matrix(self, cm):
        """
        Visualization B: Confusion Matrix Heatmap

        Args:
            cm: Confusion matrix from validation
        """
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATION: Confusion Matrix Heatmap")
        print("=" * 60)

        # Create outputs directory if it doesn't exist
        os.makedirs('outputs', exist_ok=True)

        # Create figure
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
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()
        plt.savefig('outputs/visualization_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved: outputs/visualization_confusion_matrix.png")
        plt.close()

    def visualize_feature_importance(self):
        """
        Visualization C: Feature Importance Bar Chart
        """
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATION: Feature Importance")
        print("=" * 60)

        # Create outputs directory if it doesn't exist
        os.makedirs('outputs', exist_ok=True)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Feature Coefficients
        colors = ['#3498db' if coef < 0 else '#e74c3c' for coef in self.coefficients]
        bars = ax1.barh(self.feature_names, self.coefficients, color=colors,
                        edgecolor='black', linewidth=1.5)

        ax1.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Features', fontsize=12, fontweight='bold')
        ax1.set_title('Logistic Regression Feature Coefficients',
                     fontsize=14, fontweight='bold', pad=20)
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
        ax1.grid(axis='x', alpha=0.3, linestyle='--')

        # Add value labels
        for i, (bar, coef) in enumerate(zip(bars, self.coefficients)):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{coef:.4f}',
                    ha='left' if width > 0 else 'right',
                    va='center', fontsize=10, fontweight='bold')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#e74c3c', label='Positive (→ Spam)'),
                          Patch(facecolor='#3498db', label='Negative (→ Legitimate)')]
        ax1.legend(handles=legend_elements, loc='best', fontsize=10)

        # Plot 2: Feature Correlation Heatmap
        df = pd.read_csv(self.data_path)
        correlation_matrix = df.corr()

        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, linecolor='black',
                   cbar_kws={'label': 'Correlation'},
                   ax=ax2, vmin=-1, vmax=1)

        ax2.set_title('Feature Correlation Heatmap',
                     fontsize=14, fontweight='bold', pad=20)
        ax2.set_xlabel('Features', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Features', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig('outputs/visualization_feature_importance.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved: outputs/visualization_feature_importance.png")
        plt.close()


def main():
    """
    Main function to run the email classification system
    """
    print("\n" + "=" * 60)
    print(" EMAIL SPAM CLASSIFICATION SYSTEM")
    print(" Using Logistic Regression")
    print("=" * 60)

    # Initialize classifier - UPDATED PATH
    data_path = 't_tchabukiani25_16928.csv'  # CSV file in project root
    classifier = EmailSpamClassifier(data_path)

    # Step 1 & 2: Load data and train model
    classifier.load_and_process_data()
    classifier.train_model()

    # Step 3: Validate model
    cm, accuracy = classifier.validate_model()

    # Step 7: Generate visualizations
    classifier.visualize_class_distribution()
    classifier.visualize_confusion_matrix(cm)
    classifier.visualize_feature_importance()

    # Step 4, 5, 6: Test with custom emails
    print("\n" + "=" * 60)
    print("TESTING WITH CUSTOM EMAILS")
    print("=" * 60)

    # Example Spam Email
    spam_email = """
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
    """

    print("\n" + "-" * 60)
    print("TEST 1: SPAM EMAIL EXAMPLE")
    print("-" * 60)
    print("\nEmail Text:")
    print(spam_email)
    classifier.classify_email(spam_email)

    # Example Legitimate Email
    legitimate_email = """
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
    """

    print("\n" + "-" * 60)
    print("TEST 2: LEGITIMATE EMAIL EXAMPLE")
    print("-" * 60)
    print("\nEmail Text:")
    print(legitimate_email)
    classifier.classify_email(legitimate_email)

    print("\n" + "=" * 60)
    print(" CLASSIFICATION SYSTEM COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nGenerated files:")
    print("  1. outputs/visualization_class_distribution.png")
    print("  2. outputs/visualization_confusion_matrix.png")
    print("  3. outputs/visualization_feature_importance.png")
    print("\n")


if __name__ == "__main__":
    main()
