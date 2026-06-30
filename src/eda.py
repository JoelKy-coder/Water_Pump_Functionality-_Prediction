from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Compute Cramér's V for two categorical variables."""
    contingency = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(contingency)
    n = contingency.to_numpy().sum()
    phi2 = chi2 / n
    r, k = contingency.shape
    denominator = min(r - 1, k - 1)
    return float(np.sqrt(phi2 / denominator)) if denominator > 0 else 0.0


def categorical_associations(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Return a ranked list of Cramér's V scores for categorical features."""
    rows: list[tuple[str, float]] = []
    for col in df.columns:
        if col == target_col or pd.api.types.is_numeric_dtype(df[col]):
            continue
        score = cramers_v(df[col].astype(str), df[target_col].astype(str))
        rows.append((col, float(score)))
    return pd.DataFrame(rows, columns=["feature", "cramers_v"]).sort_values("cramers_v", ascending=False)
