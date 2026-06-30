from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import HIGH_CARDINALITY_COLS, REDUNDANT_COLUMN_GROUPS, TARGET_COL
from src.utils import setup_logging

logger = setup_logging(__name__)


def normalize_boolean(series: pd.Series) -> pd.Series:
    """Normalize several boolean encodings to a pandas BooleanDtype."""
    mapping = {
        True: True,
        False: False,
        1: True,
        0: False,
        "True": True,
        "False": False,
        "true": True,
        "false": False,
        "yes": True,
        "no": False,
        "Y": True,
        "N": False,
        "Yes": True,
        "No": False,
    }
    return series.map(mapping).astype("boolean")


def clean_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning and feature engineering rules to the merged dataset."""
    df = raw_df.copy()
    logger.info("Starting cleaning pipeline")

    df = df.drop_duplicates().copy()

    for col in ["public_meeting", "permit"]:
        if col in df.columns:
            df[col] = normalize_boolean(df[col])

    if "region_code" in df.columns:
        df["region_code"] = df["region_code"].astype("category")
    if "district_code" in df.columns:
        df["district_code"] = df["district_code"].astype("category")

    if "date_recorded" in df.columns:
        df["date_recorded"] = pd.to_datetime(df["date_recorded"], errors="coerce")

    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df.loc[df[col].abs() < 1e-9, col] = np.nan

    for col in ["construction_year", "population"]:
        if col in df.columns:
            df.loc[df[col] == 0, col] = np.nan

    for col in ["funder", "installer", "subvillage", "wpt_name", "scheme_name"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

    for col in [
        "scheme_management", "water_quality", "quantity", "source", "payment", "management", "extraction_type", "waterpoint_type", "basin", "region", "lga", "ward", "recorded_by"
    ]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    for _, drop_cols in REDUNDANT_COLUMN_GROUPS.items():
        for col in drop_cols:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    for col in HIGH_CARDINALITY_COLS:
        if col in df.columns:
            top_n = 20 if col in ["funder", "installer", "scheme_name"] else 15
            counts = df[col].value_counts()
            top = set(counts.head(top_n).index)
            df[col] = df[col].where(df[col].isin(top), "Other")

    if "date_recorded" in df.columns:
        df["record_year"] = df["date_recorded"].dt.year
        df["record_month"] = df["date_recorded"].dt.month
        df["record_day"] = df["date_recorded"].dt.day
        df["record_weekday"] = df["date_recorded"].dt.dayofweek
        df.drop(columns=["date_recorded"], inplace=True)

    if "construction_year" in df.columns and "record_year" in df.columns:
        df["pump_age"] = df["record_year"] - df["construction_year"]
        df.loc[df["pump_age"] < 0, "pump_age"] = np.nan

    for col in ["recorded_by", "id"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df


def encode_target(df: pd.DataFrame, target_col: str = TARGET_COL) -> tuple[pd.DataFrame, LabelEncoder]:
    """Encode the target label so the classifier receives numeric classes."""
    if target_col not in df.columns:
        raise KeyError(f"Missing target column: {target_col}")
    encoder = LabelEncoder()
    encoded_df = df.copy()
    encoded_df[target_col] = encoder.fit_transform(encoded_df[target_col])
    return encoded_df, encoder
