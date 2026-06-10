"""
preprocess.py
Data loading, cleaning, encoding, and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

FEATURE_COLS = ["Endurance", "Strength", "Speed", "Flexibility", "Cognitive Ability", "Reflex"]
TARGET_COL = "Sport"

def load_data(path: str = "data/combined_sports_data.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[preprocess] Loaded {len(df)} rows, {df['Sport'].nunique()} sports")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    
    # Clip scores to valid range 1–10
    for col in FEATURE_COLS:
        df[col] = df[col].clip(1.0, 10.0)
    
    after = len(df)
    print(f"[preprocess] Cleaned: {before} → {after} rows ({before - after} removed)")
    return df.reset_index(drop=True)


def encode_labels(df: pd.DataFrame):
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET_COL])
    return y, le


def split_and_scale(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Returns X_train, X_test, y_train, y_test, scaler, label_encoder
    """
    df = clean_data(df)
    X = df[FEATURE_COLS].values
    y, le = encode_labels(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"[preprocess] Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"[preprocess] Sports: {list(le.classes_)}")
    return X_train, X_test, y_train, y_test, scaler, le


def save_artifacts(scaler, le, out_dir="models"):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(scaler, f"{out_dir}/scaler.pkl")
    joblib.dump(le, f"{out_dir}/label_encoder.pkl")
    print(f"[preprocess] Saved scaler + label encoder to {out_dir}/")


def load_artifacts(model_dir="models"):
    scaler = joblib.load(f"{model_dir}/scaler.pkl")
    le = joblib.load(f"{model_dir}/label_encoder.pkl")
    return scaler, le


if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test, scaler, le = split_and_scale(df)
    save_artifacts(scaler, le)
