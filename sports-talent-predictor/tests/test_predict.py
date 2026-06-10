"""
tests/test_predict.py
Run with: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import joblib
import pytest

MODELS_DIR = "models"
FEATURE_ORDER = ["Endurance", "Strength", "Speed", "Flexibility", "Cognitive Ability", "Reflex"]

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def artifacts():
    scaler = joblib.load(f"{MODELS_DIR}/scaler.pkl")
    le     = joblib.load(f"{MODELS_DIR}/label_encoder.pkl")
    rf     = joblib.load(f"{MODELS_DIR}/random_forest_model.pkl")
    ens    = joblib.load(f"{MODELS_DIR}/ensemble_model.pkl")
    return scaler, le, rf, ens


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_models_load(artifacts):
    scaler, le, rf, ens = artifacts
    assert rf is not None
    assert ens is not None
    assert len(le.classes_) == 7


def test_expected_sports(artifacts):
    _, le, _, _ = artifacts
    expected = {"Basketball", "Football", "Tennis", "Gymnastics", "LongDistance", "Wrestling", "Chess"}
    assert set(le.classes_) == expected


def test_rf_predicts_shape(artifacts):
    _, le, rf, _ = artifacts
    X = np.array([[7.0, 6.5, 7.2, 5.5, 7.0, 6.8]])
    pred = rf.predict(X)
    assert pred.shape == (1,)
    assert pred[0] in range(len(le.classes_))


def test_ensemble_predicts_shape(artifacts):
    _, le, _, ens = artifacts
    X = np.array([[7.0, 6.5, 7.2, 5.5, 7.0, 6.8]])
    pred = ens.predict(X)
    assert pred.shape == (1,)


def test_proba_sums_to_one(artifacts):
    _, le, _, ens = artifacts
    X = np.array([[7.0, 6.5, 7.2, 5.5, 7.0, 6.8]])
    proba = ens.predict_proba(X)
    assert abs(proba.sum() - 1.0) < 1e-5


def test_scores_clipped_high(artifacts):
    """Scores above 10 should still return a valid prediction (model handles clipped input)."""
    _, le, rf, _ = artifacts
    X = np.array([[12.0, 11.5, 15.0, 10.5, 9.0, 8.0]])
    X = np.clip(X, 1.0, 10.0)
    pred = rf.predict(X)
    assert pred[0] in range(len(le.classes_))


def test_scores_clipped_low(artifacts):
    """Scores at minimum should still work."""
    _, le, rf, _ = artifacts
    X = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    pred = rf.predict(X)
    assert pred[0] in range(len(le.classes_))


def test_top3_returned(artifacts):
    _, le, _, ens = artifacts
    X = np.array([[8.0, 5.0, 9.0, 4.0, 6.0, 7.5]])
    proba   = ens.predict_proba(X)[0]
    top3_idx = np.argsort(proba)[::-1][:3]
    assert len(top3_idx) == 3
    assert all(0 <= i < len(le.classes_) for i in top3_idx)


# ── Sport-profile sanity checks ───────────────────────────────────────────────

def test_endurance_athlete_gets_longdistance(artifacts):
    """High endurance + low strength → LongDistance should be in top-2."""
    _, le, _, ens = artifacts
    X = np.array([[9.5, 2.0, 5.0, 5.0, 5.0, 3.0]])
    proba    = ens.predict_proba(X)[0]
    top2_idx = np.argsort(proba)[::-1][:2]
    top2     = [le.classes_[i] for i in top2_idx]
    assert "LongDistance" in top2, f"Expected LongDistance in top-2, got {top2}"


def test_chess_profile(artifacts):
    """Very high cognitive + low physical → Chess should be in top-2."""
    _, le, _, ens = artifacts
    X = np.array([[2.0, 2.0, 2.0, 2.0, 9.5, 8.5]])
    proba    = ens.predict_proba(X)[0]
    top2_idx = np.argsort(proba)[::-1][:2]
    top2     = [le.classes_[i] for i in top2_idx]
    assert "Chess" in top2, f"Expected Chess in top-2, got {top2}"
