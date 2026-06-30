from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split numeric and categorical columns for preprocessing."""
    numeric_cols = [col for col in df.columns if col != "status_group" and pd.api.types.is_numeric_dtype(df[col])]
    categorical_cols = [col for col in df.columns if col != "status_group" and not pd.api.types.is_numeric_dtype(df[col])]
    return numeric_cols, categorical_cols


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Build a reusable preprocessing pipeline for numeric and categorical inputs."""
    numeric_cols, categorical_cols = get_feature_types(df)
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ]
    )
