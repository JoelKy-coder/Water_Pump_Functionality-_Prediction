from __future__ import annotations

import joblib
import pandas as pd

from src.config import MODELS_DIR


def load_model_bundle() -> tuple[object, object]:
    """Load the saved model bundle from disk."""
    model = joblib.load(MODELS_DIR / "best_model.pkl")
    encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")
    return model, encoder


def predict_from_row(input_row: dict[str, object]) -> str:
    """Run a single-row prediction through the trained pipeline."""
    model, encoder = load_model_bundle()
    frame = pd.DataFrame([input_row])
    prediction = model.predict(frame)[0]
    return str(encoder.inverse_transform([prediction])[0])
