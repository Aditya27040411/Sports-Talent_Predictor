"""
evaluate.py
Generates evaluation plots, SHAP explanations, Cohen's kappa, and saves results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    confusion_matrix, classification_report,
    cohen_kappa_score, accuracy_score, f1_score,
    ConfusionMatrixDisplay,
)

RESULTS_DIR = "results"
MODELS_DIR = "models"

FEATURE_COLS = ["Endurance", "Strength", "Speed", "Flexibility", "Cognitive Ability", "Reflex"]


def plot_confusion_matrix(y_true, y_pred, class_names, model_name="Model", save=True):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save:
        path = f"{RESULTS_DIR}/confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
        plt.savefig(path, dpi=150)
        print(f"[evaluate] Saved: {path}")
    plt.show()


def plot_feature_importance(model, model_name="Random Forest", save=True):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    features_sorted = [FEATURE_COLS[i] for i in indices]
    imp_sorted = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2196F3" if i == 0 else "#90CAF9" for i in range(len(features_sorted))]
    bars = ax.bar(features_sorted, imp_sorted, color=colors, edgecolor="white")
    ax.set_title(f"Feature Importances — {model_name}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Importance")
    ax.set_ylim(0, imp_sorted[0] * 1.3)
    for bar, val in zip(bars, imp_sorted):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    if save:
        path = f"{RESULTS_DIR}/feature_importance.png"
        plt.savefig(path, dpi=150)
        print(f"[evaluate] Saved: {path}")
    plt.show()


def shap_analysis(model, X_sample, class_names, save=True):
    """
    SHAP summary plot using TreeExplainer for tree-based models.
    """
    try:
        import shap
    except ImportError:
        print("[evaluate] SHAP not installed. Run: pip install shap")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Summary plot (beeswarm — shows all classes combined)
    plt.figure()
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=FEATURE_COLS,
        class_names=class_names,
        plot_type="bar",
        show=False,
    )
    plt.title("SHAP Feature Impact (mean |SHAP| across all sport classes)", fontsize=12)
    plt.tight_layout()
    if save:
        path = f"{RESULTS_DIR}/shap_summary.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[evaluate] Saved: {path}")
    plt.show()


def print_metrics(y_true, y_pred, class_names, model_name):
    acc   = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    f1    = f1_score(y_true, y_pred, average="macro")
    top3_rate = top3_hit_rate(y_true, None)  # Needs proba; see below

    print(f"\n{'─'*50}")
    print(f"  Model        : {model_name}")
    print(f"  Accuracy     : {acc:.4f}")
    print(f"  Cohen Kappa  : {kappa:.4f}  (>0.6 = good, >0.4 = moderate)")
    print(f"  Macro F1     : {f1:.4f}")
    print(f"{'─'*50}")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))


def top3_hit_rate(y_true, y_proba, top_k=3):
    """Fraction of samples where true label is in model's top-k predictions."""
    hits = 0
    for i, probs in enumerate(y_proba):
        top_k_idx = np.argsort(probs)[::-1][:top_k]
        if y_true[i] in top_k_idx:
            hits += 1
    rate = hits / len(y_true)
    print(f"[evaluate] Top-{top_k} hit rate: {rate:.4f}  ({hits}/{len(y_true)} samples)")
    return rate


def save_classification_report(y_true, y_pred, class_names, model_name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    path = f"{RESULTS_DIR}/classification_report_{model_name.replace(' ', '_').lower()}.txt"
    with open(path, "w") as f:
        f.write(f"Model: {model_name}\n{'='*50}\n")
        f.write(report)
        f.write(f"\nCohen Kappa: {cohen_kappa_score(y_true, y_pred):.4f}\n")
        f.write(f"Accuracy:    {accuracy_score(y_true, y_pred):.4f}\n")
    print(f"[evaluate] Saved: {path}")


def run_evaluation():
    from src.preprocess import load_data, split_and_scale

    df = load_data()
    X_train, X_test, y_train, y_test, scaler, le = split_and_scale(df)
    class_names = list(le.classes_)

    # Raw (unscaled) test split for RF
    from src.preprocess import clean_data, encode_labels, FEATURE_COLS as FC
    from sklearn.model_selection import train_test_split
    df_clean = clean_data(df)
    X_raw = df_clean[FC].values
    y_raw, _ = encode_labels(df_clean)
    _, X_raw_test, _, y_raw_test = train_test_split(
        X_raw, y_raw, test_size=0.2, stratify=y_raw, random_state=42
    )

    for model_name, fname, X_t, y_t in [
        ("Random Forest",    "random_forest_model.pkl", X_raw_test, y_raw_test),
        ("SVM",              "svm_model.pkl",           X_test,     y_test),
        ("Stacked Ensemble", "ensemble_model.pkl",      X_raw_test, y_raw_test),
    ]:
        model = joblib.load(f"{MODELS_DIR}/{fname}")
        y_pred = model.predict(X_t)
        y_proba = model.predict_proba(X_t)

        print_metrics(y_t, y_pred, class_names, model_name)
        top3_hit_rate(y_t, y_proba)
        plot_confusion_matrix(y_t, y_pred, class_names, model_name)
        save_classification_report(y_t, y_pred, class_names, model_name)

    # SHAP on Random Forest
    rf = joblib.load(f"{MODELS_DIR}/random_forest_model.pkl")
    shap_analysis(rf, X_raw_test[:100], class_names)

    # Feature importance
    plot_feature_importance(rf)


if __name__ == "__main__":
    run_evaluation()
