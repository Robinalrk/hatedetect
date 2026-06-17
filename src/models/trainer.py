"""
Model Development & Diagnostic Validation module.
Trains multiple classifiers with SMOTE, evaluates using Confusion Matrix,
Precision, Recall, and F1-Score.
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from src.preprocessing.cleaner import TextCleaner


class ModelTrainer:
    """Trains, evaluates, and saves content moderation models."""

    def __init__(self, data_path: str = "labeled_data.csv"):
        self.data_path = data_path
        self.cleaner = TextCleaner()
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.models = {}
        self.results = {}
        self.X_test = None
        self.y_test = None

    def load_and_preprocess(self):
        """Load dataset, clean text, extract features."""
        df = pd.read_csv(self.data_path)
        df = df.drop(columns=['Unnamed: 0', 'count', 'hate_speech', 'offensive_language', 'neither'],
                     errors='ignore')
        df['cleaned_tweet'] = df['tweet'].apply(self.cleaner.clean)
        X = self.vectorizer.fit_transform(df['cleaned_tweet']).toarray()
        y = df['class'].values
        return X, y

    def train_all(self):
        """Train multiple classifiers with SMOTE and evaluate."""
        print("=" * 60)
        print("CONTENT MODERATION - MODEL TRAINING")
        print("=" * 60)

        X, y = self.load_and_preprocess()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.X_test = X_test
        self.y_test = y_test

        # Handle class imbalance with SMOTE
        print("\nApplying SMOTE to balance classes...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        print(f"  Original: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        print(f"  Resampled: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")

        # 1. ComplementNB - optimized for imbalanced text
        print("\n[1/3] Training ComplementNB...")
        nb = ComplementNB()
        nb.fit(X_resampled, y_resampled)
        y_pred = nb.predict(X_test)
        self._log_results("ComplementNB", y_test, y_pred)
        self.models["ComplementNB"] = nb

        # 2. LogisticRegression
        print("\n[2/3] Training LogisticRegression...")
        lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        lr.fit(X_resampled, y_resampled)
        y_pred = lr.predict(X_test)
        self._log_results("LogisticRegression", y_test, y_pred)
        self.models["LogisticRegression"] = lr

        # 3. RandomForestClassifier
        print("\n[3/3] Training RandomForestClassifier...")
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        rf.fit(X_resampled, y_resampled)
        y_pred = rf.predict(X_test)
        self._log_results("RandomForest", y_test, y_pred)
        self.models["RandomForest"] = rf

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        return self.results

    def _log_results(self, name: str, y_test, y_pred):
        """Log evaluation metrics for a model."""
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        accuracy = accuracy_score(y_test, y_pred)
        self.results[name] = {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Classification Report:\n{classification_report(y_test, y_pred)}")

    def save_best_model(self, metric: str = "accuracy"):
        """Save the best performing model and vectorizer."""
        best_name = max(self.results, key=lambda k: self.results[k][metric])
        best_model = self.models[best_name]
        print(f"\nSaving best model: {best_name}")

        with open('model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        with open('tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)

        # Save test data for API metrics
        with open('test_data.pkl', 'wb') as f:
            pickle.dump({
                'y_test': self.y_test,
                'y_pred': best_model.predict(self.X_test),
                'model_name': best_name,
            }, f)
        print(f"  ✓ Model saved to models_saved/")
        return best_name