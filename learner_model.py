import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "student_risk_model.joblib"
METRICS_PATH = ROOT / "models" / "evaluation_metrics.json"


def load_trained_model():
    """Load the fitted UCI model, or return None until training has run."""
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except (OSError, ValueError, EOFError):
        return None


def predict_student_risk(features: dict):
    """
    Predict the UCI dataset's historical school-risk label.
    This is not a predictor for StudyPath technical quiz results.
    """
    model = load_trained_model()
    if model is None:
        return None
    row = pd.DataFrame([features])
    prediction = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    classes = list(model.classes_)
    risk_index = classes.index(1) if 1 in classes else None
    risk_probability = float(probabilities[risk_index]) if risk_index is not None else 0.0
    return {
        "prediction": prediction,
        "label": "At risk in historical course data" if prediction == 1 else "Not flagged as at risk",
        "risk_probability": round(risk_probability, 4),
        "note": "Educational demonstration only; not a high-stakes decision tool.",
    }


def read_evaluation_metrics():
    if not METRICS_PATH.exists():
        return None
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
