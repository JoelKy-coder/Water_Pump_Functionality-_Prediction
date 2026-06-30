import logging
from pathlib import Path


def setup_logging(name: str = "water_pump") -> logging.Logger:
    """Configure and return a module logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(levelname)s - %(name)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def ensure_parent_dir(path: Path) -> None:
    """Create parent directories for a file path if absent."""
    path.parent.mkdir(parents=True, exist_ok=True)
