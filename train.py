"""Train a spam email classifier using TF-IDF and Naive Bayes."""
from datetime import datetime
import json
import os

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from data import get_dataset_stats, load_dataset


def build_model():
    """Create the TF-IDF + Multinomial Naive Bayes pipeline with unigrams and bigrams."""
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=5000,
                sublinear_tf=True,
            ),
        ),
        ("clf", MultinomialNB(alpha=0.1)),
    ])


def cross_validate_model(texts, labels, n_folds=5):
    """Run stratified k-fold cross-validation and return per-fold accuracy scores."""
    model = build_model()
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, texts, labels, cv=cv, scoring="accuracy")
    return scores


def extract_top_features(model, top_k=20):
    """Extract the top keywords contributing most strongly to spam and ham classifications."""
    tfidf = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    feature_names = tfidf.get_feature_names_out()

    # clf.classes_: typically [0, 1] for [ham, spam]
    classes = list(clf.classes_)
    ham_idx = classes.index(0)
    spam_idx = classes.index(1)

    ham_log_probs = clf.feature_log_prob_[ham_idx]
    spam_log_probs = clf.feature_log_prob_[spam_idx]

    # Ratio of log-probabilities
    log_ratio = spam_log_probs - ham_log_probs
    sorted_spam_indices = np.argsort(log_ratio)[::-1]
    sorted_ham_indices = np.argsort(log_ratio)

    top_spam = [
        {"word": str(feature_names[i]), "score": float(log_ratio[i])}
        for i in sorted_spam_indices[:top_k]
    ]
    top_ham = [
        {"word": str(feature_names[i]), "score": float(-log_ratio[i])}
        for i in sorted_ham_indices[:top_k]
    ]

    return top_spam, top_ham


def main():
    print("=" * 60)
    print("  Training Spam Email Classifier")
    print("=" * 60)

    # 1. Load labeled dataset (with text preprocessing applied)
    texts, labels = load_dataset()
    stats = get_dataset_stats(texts, labels)
    print(f"Loaded {stats['total']} samples: {stats['ham_count']} ham ({stats['ham_pct']:.1f}%), {stats['spam_count']} spam ({stats['spam_pct']:.1f}%)")

    # 2. 5-fold Stratified Cross-Validation for robust accuracy estimate
    print("\nRunning 5-fold stratified cross-validation...")
    cv_scores = cross_validate_model(texts, labels)
    print(f"CV Fold Scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"CV Mean Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 3. Stratified train/test split for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"\nTraining set: {len(X_train)} samples | Test set: {len(X_test)} samples")

    # 4. Build and train final model pipeline
    model = build_model()
    model.fit(X_train, y_train)

    # 5. Comprehensive Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_text = classification_report(y_test, y_pred, target_names=["Ham", "Spam"])
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"\nTest Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    print("\nClassification Report:")
    print(report_text)
    print("Confusion Matrix:")
    print(np.array(cm))

    # 6. Extract top features
    top_spam, top_ham = extract_top_features(model)

    # 7. Save model with joblib (recommended over pickle for sklearn models)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model.pkl")
    joblib.dump(model, model_path)
    print(f"\n[OK] Model successfully saved to {model_path} (via joblib)")

    metadata = {
        "task": "Spam Email Classifier",
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy": float(acc),
        "cross_validation": {
            "n_folds": 5,
            "fold_scores": [float(s) for s in cv_scores],
            "mean_accuracy": float(cv_scores.mean()),
            "std_accuracy": float(cv_scores.std()),
        },
        "dataset_stats": stats,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "confusion_matrix": cm,
        "classification_report": report_dict,
        "top_spam_keywords": top_spam,
        "top_ham_keywords": top_ham,
    }

    metadata_path = os.path.join(base_dir, "model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Training metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()