import json
import time

from src.config import PROCESSED_DIR, TARGET_COL
from src.data_loader import load_merged_data
from src.evaluation import compute_metrics, save_artifacts
from src.model_training import train_and_select_model
from src.preprocessing import clean_dataframe, encode_target
from src.utils import setup_logging

logger = setup_logging(__name__)


def main() -> None:
    """Run the full end-to-end training pipeline."""
    logger.info("Starting pipeline")
    raw_df = load_merged_data()
    cleaned_df = clean_dataframe(raw_df)
    cleaned_df.to_csv(PROCESSED_DIR / "train_clean.csv", index=False)
    encoded_df, encoder = encode_target(cleaned_df)
    _, metadata, model = train_and_select_model(encoded_df)
    X = encoded_df.drop(columns=[TARGET_COL])
    y = encoded_df[TARGET_COL]
    predictions = model.predict(X)
    metrics = compute_metrics(y, predictions)
    metadata.update(
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "feature_list": X.columns.tolist(),
            "metrics": metrics,
        }
    )
    save_artifacts(model, encoder, metadata)
    logger.info("Pipeline completed with metrics: %s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
