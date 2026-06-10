"""
gan_augment.py
Synthetic data augmentation using CTGAN (Conditional Tabular GAN).
Generates balanced synthetic samples per sport class.

Install:   pip install ctgan
"""

import os
import pandas as pd
import numpy as np

FEATURE_COLS = ["Endurance", "Strength", "Speed", "Flexibility", "Cognitive Ability", "Reflex"]
TARGET_COL   = "Sport"


def augment_with_ctgan(
    df: pd.DataFrame,
    target_per_class: int = 300,
    epochs: int = 300,
    batch_size: int = 500,
    save_path: str = "data/augmented_sports_data.csv",
) -> pd.DataFrame:
    """
    Uses CTGAN to generate synthetic rows until each sport class has
    `target_per_class` samples, then appends to original data.
    """
    try:
        from ctgan import CTGAN
    except ImportError:
        raise ImportError("Install CTGAN: pip install ctgan")

    all_synthetic = []
    sports = df[TARGET_COL].unique()

    for sport in sports:
        sport_df = df[df[TARGET_COL] == sport].reset_index(drop=True)
        current_n = len(sport_df)
        need = max(0, target_per_class - current_n)

        if need == 0:
            print(f"[gan] {sport}: already has {current_n} ≥ {target_per_class}, skipping")
            continue

        print(f"[gan] {sport}: {current_n} rows → generating {need} synthetic …")

        # Train CTGAN on this sport's data only
        model = CTGAN(epochs=epochs, batch_size=batch_size, verbose=False)
        model.fit(sport_df[FEATURE_COLS])

        synth = model.sample(need)
        synth[TARGET_COL] = sport
        # Clip to valid 1–10 range
        synth[FEATURE_COLS] = synth[FEATURE_COLS].clip(1.0, 10.0)
        all_synthetic.append(synth)

    if all_synthetic:
        synth_df = pd.concat(all_synthetic, ignore_index=True)
        combined = pd.concat([df, synth_df], ignore_index=True).sample(frac=1, random_state=42)
        print(f"\n[gan] Original: {len(df)} → Augmented: {len(combined)} rows")
    else:
        combined = df
        print("[gan] No augmentation needed.")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    combined.to_csv(save_path, index=False)
    print(f"[gan] Saved augmented dataset: {save_path}")
    return combined


def validate_augmentation(df_real: pd.DataFrame, df_synth: pd.DataFrame):
    """
    Quick statistical validation: KS test + real-vs-synthetic classifier AUC.
    Prints per-feature KS p-values and classifier AUC.
    """
    from scipy.stats import ks_2samp
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    print("\n[validate] ── Feature-wise KS Test (real vs synthetic) ──")
    for col in FEATURE_COLS:
        r = df_real[col].dropna().values
        s = df_synth[col].dropna().values
        stat, p = ks_2samp(r, s)
        status = "✅" if p > 0.05 else "⚠️ "
        print(f"  {status} {col:<22s}  KS p = {p:.4f}")

    print("\n[validate] ── Real-vs-Synthetic Classifier (AUC near 0.5 = good) ──")
    X = pd.concat([df_real[FEATURE_COLS], df_synth[FEATURE_COLS]], ignore_index=True)
    y = np.hstack([np.ones(len(df_real)), np.zeros(len(df_synth))])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
    quality = "✅ Excellent" if auc < 0.55 else ("⚠️  Acceptable" if auc < 0.70 else "❌ Needs review")
    print(f"  Classifier AUC: {auc:.4f}  → {quality}")


if __name__ == "__main__":
    df = pd.read_csv("data/combined_sports_data.csv").dropna()
    augmented = augment_with_ctgan(df, target_per_class=300)

    # Validate quality of synthetic data
    synth_only = augmented.iloc[len(df):]
    validate_augmentation(df, synth_only)
