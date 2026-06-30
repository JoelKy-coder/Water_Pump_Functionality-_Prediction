from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RANDOM_STATE = 42
TARGET_COL = "status_group"
TEST_SIZE = 0.2
CV_FOLDS = 5
TARGET_CLASSES = ["functional", "functional needs repair", "non functional"]
HIGH_CARDINALITY_COLS = ["funder", "installer", "subvillage", "wpt_name", "scheme_name"]
REDUNDANT_COLUMN_GROUPS = {
    "extraction_type": ["extraction_type_group", "extraction_type_class"],
    "management": ["management_group"],
    "payment": ["payment_type"],
    "water_quality": ["quality_group"],
    "quantity": ["quantity_group"],
    "source": ["source_type", "source_class"],
    "waterpoint_type": ["waterpoint_type_group"],
}
