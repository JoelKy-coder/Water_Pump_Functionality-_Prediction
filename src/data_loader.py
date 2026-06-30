from pathlib import Path

import pandas as pd

from src.config import RAW_DIR
from src.utils import setup_logging

logger = setup_logging(__name__)


def load_features(path: Path | None = None) -> pd.DataFrame:
    """Load raw features data from CSV."""
    file_path = path or RAW_DIR / "features.csv"
    logger.info("Loading features from %s", file_path)
    return pd.read_csv(file_path)


def load_labels(path: Path | None = None) -> pd.DataFrame:
    """Load raw labels data from CSV."""
    file_path = path or RAW_DIR / "labels.csv"
    logger.info("Loading labels from %s", file_path)
    return pd.read_csv(file_path)


def load_merged_data(features_path: Path | None = None, labels_path: Path | None = None) -> pd.DataFrame:
    """Load and merge raw features with labels using the id key."""
    features = load_features(features_path)
    labels = load_labels(labels_path)
    merged = pd.merge(features, labels, on="id", how="inner")
    logger.info("Merged %d rows with labels", len(merged))
    return merged
