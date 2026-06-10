"""
train.py
Trains Random Forest, SVM, and a Stacked Ensemble.
Saves all models and prints CV results.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, make_scorer, f1_score
from src.preprocess import load_data, split_and_scale, save_artifacts

MODELS_DIR = "models"
DATA_PATH = "data/combined_sports_data.csv"


# ────────────────────────────────────────────────
# 1. Model Definitions
# ────────────────────────────────────────────────

def build_random_forest():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def build_svm():
    # SVM needs probability=True to output confidence scores
    return SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=42,
    )


def build_gradient_boosting():
    return GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        random_state=42,
    )


def build_stacked_ensemble(rf, svm, gb):
    """
    Level-0: RF + SVM + GB
    Level-1 meta-learner: Logistic Regression with CV predictions
    """
    estimators = [
        ("random_forest", rf),
        ("svm", svm),
        ("gradient_boosting", gb),
    ]
    meta = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    return StackingClassifier(
        estimators=estimators,
        final_estimator=meta,
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )


# ────────────────────────────────────────────────
# 2. Cross-Validation
# ────────────────────────────────────────────────

def cross_validate_model(model, X, y, name: str, cv_folds: int = 5):
    print(f"\n[train] ── Cross-validating: {name} ({cv_folds}-fold) ──")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "macro_f1": make_scorer(f1_score, average="macro"),
        "weighted_f1": make_scorer(f1_score, average="weighted"),
    }
    results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    for metric, scores in results.items():
        if metric.startswith("test_"):
            label = metric[5:]
            print(f"  {label:>12s}: {scores.mean():.4f} ± {scores.std():.4f}")
    return results


# ────────────────────────────────────────────────
# 3. Train, Evaluate, Save
# ────────────────────────────────────────────────

def train_all(data_path: str = DATA_PATH):
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Load & preprocess ──
    df = load_data(data_path)
    X_train, X_test, y_train, y_test, scaler, le = split_and_scale(df)
    save_artifacts(scaler, le, MODELS_DIR)

    # Keep unscaled for RF (trees don't need scaling)
    from src.preprocess import clean_data, encode_labels, FEATURE_COLS
    df_clean = clean_data(df)
    X_raw = df_clean[FEATURE_COLS].values
    y_raw, _ = encode_labels(df_clean)
    from sklearn.model_selection import train_test_split
    X_raw_train, X_raw_test, yr_train, yr_test = train_test_split(
        X_raw, y_raw, test_size=0.2, stratify=y_raw, random_state=42
    )

    # ── Build models ──
    rf  = build_random_forest()
    svm = build_svm()
    gb  = build_gradient_boosting()

    # ── Cross-validation on full dataset ──
    cross_validate_model(rf,  X_raw, y_raw, "Random Forest")
    cross_validate_model(svm, scaler.transform(X_raw), y_raw, "SVM (RBF)")
    cross_validate_model(gb,  X_raw, y_raw, "Gradient Boosting")

    # ── Final training on train split ──
    print("\n[train] Training final models on train split …")
    rf.fit(X_raw_train, yr_train)
    svm.fit(X_train, y_train)
    gb.fit(X_raw_train, yr_train)

    # Stacked ensemble (uses scaled features for all base models via pipeline)
    print("[train] Training stacked ensemble …")
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler as SS

    rf_pipe  = Pipeline([("rf",  build_random_forest())])
    svm_pipe = Pipeline([("sc",  SS()), ("svm", build_svm())])
    gb_pipe  = Pipeline([("gb",  build_gradient_boosting())])
    ensemble = StackingClassifier(
        estimators=[("rf", rf_pipe), ("svm", svm_pipe), ("gb", gb_pipe)],
        final_estimator=LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        cv=5, stack_method="predict_proba", n_jobs=-1,
    )
    ensemble.fit(X_raw_train, yr_train)

    # ── Test-set evaluation ──
    print("\n[train] ── Test-set Results ──")
    models_to_eval = {
        "Random Forest":      (rf,       X_raw_test),
        "SVM":                (svm,      X_test),
        "Gradient Boosting":  (gb,       X_raw_test),
        "Stacked Ensemble":   (ensemble, X_raw_test),
    }
    for name, (m, Xt) in models_to_eval.items():
        preds = m.predict(Xt)
        report = classification_report(yr_test, preds, target_names=le.classes_, digits=3)
        print(f"\n  ── {name} ──\n{report}")

    # ── Save ──
    print("\n[train] Saving models …")
    joblib.dump(rf,       f"{MODELS_DIR}/random_forest_model.pkl")
    joblib.dump(svm,      f"{MODELS_DIR}/svm_model.pkl")
    joblib.dump(gb,       f"{MODELS_DIR}/gradient_boosting_model.pkl")
    joblib.dump(ensemble, f"{MODELS_DIR}/ensemble_model.pkl")
    print("[train] Done. All models saved to models/")

    return rf, svm, gb, ensemble, scaler, le, (X_raw_test, yr_test, X_test)


if __name__ == "__main__":
    train_all()
