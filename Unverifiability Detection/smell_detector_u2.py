import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
import joblib
import os
import warnings
import re
from sklearn.base import BaseEstimator, TransformerMixin
import json
import optuna

warnings.filterwarnings('ignore')


# =====================================================
# GLOBAL WEIGHT TRANSFORMER
# =====================================================
class WeightApplier(BaseEstimator, TransformerMixin):
    def __init__(self, weight=1.0):
        self.weight = weight

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X * self.weight


# =====================================================
# 1. DATA LOADING
# =====================================================
def load_data():
    while True:
        csv_path = input("Enter dataset CSV path: ").strip()
        if not os.path.exists(csv_path):
            print(f"Error: File not found at {csv_path}")
            continue
        
        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.lower().str.strip()
            
            text_col = next((c for c in df.columns if 'req' in c or 'text' in c), None)
            label_col = next((c for c in df.columns if 'unverifiability' in c or 'label' in c or 'target' in c), None)
            
            if text_col is None or label_col is None:
                raise ValueError("Text or label column not found.")
            
            df = df.rename(columns={text_col: 'text', label_col: 'label'})
            df['text'] = df['text'].astype(str).str.strip()
            df['label'] = df['label'].astype(int)
            
            print(f"\nLoaded {len(df)} records.")
            print("Class distribution:\n", df['label'].value_counts(normalize=True))
            return df
        
        except Exception as e:
            print(f"Data loading error: {str(e)}")


# =====================================================
# 2. FEATURE ENGINEERING
# =====================================================
class EnhancedFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = pd.Series(X['text']) if isinstance(X, pd.DataFrame) else X
        
        features = pd.DataFrame({
            'length': X.str.len(),
            'num_words': X.str.split().str.len(),
            'num_sentences': X.str.count(r'[.!?]'),
            'modals': X.str.count(r'\b(can|could|may|might|must|shall|should|will|would)\b', flags=re.IGNORECASE),
            'vague_terms': X.str.count(r'\b(some|various|many|several|appropriate|user-friendly)\b', flags=re.IGNORECASE),
            'adverbs': X.str.count(r'\b(quickly|easily|significantly|approximately)\b', flags=re.IGNORECASE),
            'comparatives': X.str.count(r'\b(better|faster|cheaper|improved)\b', flags=re.IGNORECASE),
            'superlatives': X.str.count(r'\b(best|fastest|most|least)\b', flags=re.IGNORECASE),
            'long_sentence': (X.str.split().str.len() > 25).astype(int),
            'avg_word_len': X.apply(lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0)
        }).fillna(0)
        
        return features.values


# =====================================================
# 3. WEIGHTED PIPELINE
# =====================================================
def build_pipeline(n_estimators, max_depth, min_samples_leaf):
    word_tfidf_weight = 1.0
    char_tfidf_weight = 0.6
    numeric_feature_weight = 0.4

    column_features = ColumnTransformer(
        transformers=[
            ('tfidf_word', Pipeline([
                ('vec', TfidfVectorizer(
                    max_features=8000,
                    ngram_range=(1, 3),
                    stop_words='english',
                    sublinear_tf=True)),
                ('weight', WeightApplier(word_tfidf_weight))
            ]), 'text'),

            ('tfidf_char', Pipeline([
                ('vec', TfidfVectorizer(
                    max_features=3000,
                    ngram_range=(3, 6),
                    analyzer='char_wb')),
                ('weight', WeightApplier(char_tfidf_weight))
            ]), 'text'),

            ('numeric', Pipeline([
                ('features', EnhancedFeatureExtractor()),
                ('scaler', StandardScaler()),
                ('weight', WeightApplier(numeric_feature_weight))
            ]), 'text')
        ],
        remainder='drop'
    )

    return Pipeline([
        ('features', column_features),
        ('classifier', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        ))
    ])


# =====================================================
# 4. OPTUNA OBJECTIVE
# =====================================================
def objective(trial, X_train, y_train, X_valid, y_valid):
    n_estimators = trial.suggest_int("n_estimators", 200, 700)
    max_depth = trial.suggest_int("max_depth", 8, 25)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)

    model = build_pipeline(n_estimators, max_depth, min_samples_leaf)
    model.fit(X_train, y_train)

    preds = model.predict(X_valid)
    report = classification_report(y_valid, preds, output_dict=True)

    return report["weighted avg"]["f1-score"]


# =====================================================
# 5. OPTIMIZED-ONLY EVALUATION
# =====================================================
def evaluate_and_save(model, X_test, y_test):

    calibrated = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
    calibrated.fit(X_test, y_test)

    y_probs = calibrated.predict_proba(X_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_threshold = thresholds[np.argmax(f1_scores)]

    y_opt = (y_probs >= optimal_threshold).astype(int)
    report_opt = classification_report(y_test, y_opt, output_dict=True)
    cm_opt = confusion_matrix(y_test, y_opt)

    print("\n=== FINAL OPTIMIZED RESULTS ===")
    print(classification_report(y_test, y_opt))
    print("Confusion Matrix:\n", cm_opt)

    return calibrated, optimal_threshold, report_opt, cm_opt


# =====================================================
# 6. MAIN DRIVER
# =====================================================
def main():
    print("\n=== Unverifiability Detector (Weighted + Optuna, Optimized Only) ===")
    df = load_data()

    X = df[['text']]
    y = df['label']

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.25, stratify=y
    )

    print("\nRunning Optuna hyperparameter search...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, X_train, y_train, X_valid, y_valid), n_trials=20)

    best_params = study.best_params
    print("\nBest Hyperparameters:", best_params)

    print("\nTraining final model with best parameters...")
    final_model = build_pipeline(
        best_params["n_estimators"],
        best_params["max_depth"],
        best_params["min_samples_leaf"]
    )
    final_model.fit(X_train, y_train)

    # FIRST display optimized results
    calibrated_model, threshold, report_opt, cm_opt = evaluate_and_save(
        final_model, X_valid, y_valid
    )

    # NOW ask where to save results
    save_path = input("\nSave directory (Enter for ./unverifiability_results): ").strip() or "./unverifiability_results"
    os.makedirs(save_path, exist_ok=True)

    # Save models
    joblib.dump(final_model, os.path.join(save_path, "model_optuna.joblib"))
    joblib.dump(calibrated_model, os.path.join(save_path, "calibrated_optuna.joblib"))

    # Save JSON results
    results_json = {
        'evaluation': report_opt,
        'optimal_threshold': float(threshold)
    }
    with open(os.path.join(save_path, "unverifiability_results.json"), "w") as f:
        json.dump(results_json, f, indent=4)

    print("\nSaved models and results to:", save_path)
    print(f"Optimal decision threshold: {threshold:.3f}")


if __name__ == "__main__":
    main()
