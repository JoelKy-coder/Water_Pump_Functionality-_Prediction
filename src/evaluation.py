from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.config import MODELS_DIR
from src.utils import ensure_parent_dir, setup_logging

logger = setup_logging(__name__)


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Compute common multiclass classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def save_artifacts(model, encoder, metadata: dict, model_path: Path | None = None, encoder_path: Path | None = None, metadata_path: Path | None = None) -> None:
    """Persist the trained model and metadata for inference."""
    model_path = model_path or MODELS_DIR / "best_model.pkl"
    encoder_path = encoder_path or MODELS_DIR / "label_encoder.pkl"
    metadata_path = metadata_path or MODELS_DIR / "metadata.json"
    ensure_parent_dir(model_path)
    ensure_parent_dir(encoder_path)
    ensure_parent_dir(metadata_path)
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    logger.info("Saved artifacts to %s", MODELS_DIR)
