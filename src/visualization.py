from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIGURES_DIR
from src.utils import ensure_parent_dir


def save_plot(fig, filename: str) -> Path:
    """Save a figure to the reports directory and close the figure handle."""
    path = FIGURES_DIR / filename
    ensure_parent_dir(path)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_target_distribution(df, target_col: str, filename: str = "target_distribution.png"):
    """Create and save a target distribution plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x=target_col, order=df[target_col].value_counts().index, ax=ax)
    ax.set_title("Target distribution")
    ax.set_xlabel(target_col)
    ax.set_ylabel("Count")
    return save_plot(fig, filename)
