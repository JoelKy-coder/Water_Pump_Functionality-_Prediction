from __future__ import annotations

import importlib
import time

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import CV_FOLDS, RANDOM_STATE, TARGET_COL, TEST_SIZE
from src.feature_engineering import build_preprocessor
from src.utils import setup_logging

logger = setup_logging(__name__)


def build_model_registry() -> dict[str, Pipeline]:
    """Build a registry of candidate classifier pipelines."""
    registry: dict[str, Pipeline] = {
        "logistic_regression": Pipeline([("classifier", LogisticRegression(max_iter=1200, random_state=RANDOM_STATE, n_jobs=-1))]),
        "decision_tree": Pipeline([("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=12))]),
        "random_forest": Pipeline([("classifier", RandomForestClassifier(n_estimators=60, random_state=RANDOM_STATE, n_jobs=-1, max_depth=12))]),
        "gradient_boosting": Pipeline([("classifier", GradientBoostingClassifier(random_state=RANDOM_STATE, n_estimators=80, max_depth=3))]),
        "extra_trees": Pipeline([("classifier", ExtraTreesClassifier(n_estimators=80, random_state=RANDOM_STATE, n_jobs=-1, max_depth=12))]),
    }
    try:
        xgb = importlib.import_module("xgboost")
        registry["xgboost"] = Pipeline([("classifier", xgb.XGBClassifier(n_estimators=50, random_state=RANDOM_STATE, objective="multi:softprob", eval_metric="mlogloss", n_jobs=1))])
    except Exception:
        logger.warning("XGBoost is not available. Skipping XGBoost benchmark.")
    return registry


def train_and_select_model(df: pd.DataFrame) -> tuple[dict, dict, Pipeline]:
    """Train candidate models and select the best pipeline by CV weighted F1."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    registry = build_model_registry()
    results: dict[str, dict] = {}
    for name, pipeline in registry.items():
        full_pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(X_train)),
                ("classifier", pipeline.named_steps["classifier"]),
            ]
        )
        start = time.time()
        scores = cross_val_score(full_pipeline, X_train, y_train, cv=CV_FOLDS, scoring="f1_weighted")
        full_pipeline.fit(X_train, y_train)
        results[name] = {
            "cv_weighted_f1_mean": float(scores.mean()),
            "cv_weighted_f1_std": float(scores.std()),
            "train_time_sec": round(time.time() - start, 3),
        }
        logger.info("Model %s mean CV weighted F1: %.4f", name, scores.mean())

    best_name = max(results, key=lambda name: results[name]["cv_weighted_f1_mean"])
    best_model = registry[best_name]
    best_pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", best_model.named_steps["classifier"]),
        ]
    )
    best_pipeline.fit(X_train, y_train)
    metadata = {
        "best_model_name": best_name,
        "cv_results": results,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "feature_count": int(X_train.shape[1]),
    }
    logger.info("Selected best model: %s", best_name)
    return results, metadata, best_pipeline
