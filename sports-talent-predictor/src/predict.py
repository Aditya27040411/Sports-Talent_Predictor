"""
predict.py
Single-child prediction interface.
Usage:
  python src/predict.py --speed 7.5 --strength 6.0 --endurance 8.2 \
                         --flexibility 5.5 --cognitive 7.0 --reflex 6.8
"""

import argparse
import numpy as np
import joblib
import os

MODELS_DIR = "models"
FEATURE_ORDER = ["Endurance", "Strength", "Speed", "Flexibility", "Cognitive Ability", "Reflex"]

SPORT_EMOJIS = {
    "Basketball": "🏀",
    "Football": "⚽",
    "Tennis": "🎾",
    "Gymnastics": "🤸",
    "LongDistance": "🏃",
    "Wrestling": "🤼",
    "Chess": "♟️",
}

SPORT_DESCRIPTIONS = {
    "Basketball": "Demands explosive speed, coordination, and high cognitive decision-making.",
    "Football": "Requires a mix of speed, team strategy, and sustained endurance.",
    "Tennis": "Excellent reflex, agility, and tactical thinking are critical.",
    "Gymnastics": "Outstanding flexibility, body control, and strength-to-weight ratio.",
    "LongDistance": "Exceptional aerobic capacity and mental resilience.",
    "Wrestling": "Dominant in strength, leverage, and close-combat endurance.",
    "Chess": "High cognitive ability and pattern-recognition speed are paramount.",
}


def load_model_and_artifacts(model_name="ensemble"):
    model_files = {
        "rf":       "random_forest_model.pkl",
        "svm":      "svm_model.pkl",
        "ensemble": "ensemble_model.pkl",
    }
    fname = model_files.get(model_name, "ensemble_model.pkl")
    model  = joblib.load(os.path.join(MODELS_DIR, fname))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    le     = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    return model, scaler, le


def predict_sport(scores: dict, model_name: str = "ensemble", top_k: int = 3) -> list:
    """
    scores: dict with keys matching FEATURE_ORDER
    Returns: list of (sport_name, confidence_%) tuples, sorted by confidence
    """
    model, scaler, le = load_model_and_artifacts(model_name)

    # Build feature vector in the correct order
    x = np.array([[scores[f] for f in FEATURE_ORDER]])

    # SVM was trained on scaled features; RF/ensemble on raw
    if model_name == "svm":
        x = scaler.transform(x)

    proba = model.predict_proba(x)[0]
    top_idx = np.argsort(proba)[::-1][:top_k]

    results = [(le.classes_[i], round(proba[i] * 100, 1)) for i in top_idx]
    return results


def print_report(scores: dict, recommendations: list):
    print("\n" + "═" * 52)
    print("  🏅  SPORTTALENT AI — Child Aptitude Report")
    print("═" * 52)
    print("\n📊 Input Profile:")
    for feat, val in scores.items():
        bar = "█" * int(val) + "░" * (10 - int(val))
        print(f"  {feat:<20s} [{bar}] {val:.1f}/10")

    print("\n🎯 Top Sport Recommendations:\n")
    medal = ["🥇", "🥈", "🥉"]
    for i, (sport, conf) in enumerate(recommendations):
        emoji = SPORT_EMOJIS.get(sport, "🏅")
        desc  = SPORT_DESCRIPTIONS.get(sport, "")
        print(f"  {medal[i]}  {emoji}  {sport:<20s}  (Confidence: {conf:.1f}%)")
        print(f"       └─ {desc}\n")

    print("═" * 52)
    print("  ⚠️  Recommendations are probabilistic guides, not verdicts.")
    print("     Reassess at age 12–14 as the child develops.")
    print("═" * 52 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Predict best sports for a child (scores 1–10)")
    parser.add_argument("--speed",       type=float, required=True)
    parser.add_argument("--strength",    type=float, required=True)
    parser.add_argument("--endurance",   type=float, required=True)
    parser.add_argument("--flexibility", type=float, required=True)
    parser.add_argument("--cognitive",   type=float, required=True)
    parser.add_argument("--reflex",      type=float, required=True)
    parser.add_argument("--model",       type=str, default="ensemble",
                        choices=["rf", "svm", "ensemble"],
                        help="Which model to use for prediction")
    parser.add_argument("--top_k",       type=int, default=3)
    args = parser.parse_args()

    scores = {
        "Endurance":        args.endurance,
        "Strength":         args.strength,
        "Speed":            args.speed,
        "Flexibility":      args.flexibility,
        "Cognitive Ability": args.cognitive,
        "Reflex":           args.reflex,
    }

    recommendations = predict_sport(scores, model_name=args.model, top_k=args.top_k)
    print_report(scores, recommendations)


if __name__ == "__main__":
    main()
