import pandas as pd

from src.data_loader import load_merged_data


def test_load_merged_data_shape_and_columns(tmp_path):
    features = pd.DataFrame({"id": [1, 2], "amount_tsh": [1.0, 2.0]})
    labels = pd.DataFrame({"id": [1, 2], "status_group": ["functional", "non functional"]})
    features_path = tmp_path / "features.csv"
    labels_path = tmp_path / "labels.csv"
    features.to_csv(features_path, index=False)
    labels.to_csv(labels_path, index=False)

    merged = load_merged_data(features_path, labels_path)

    assert merged.shape == (2, 3)
    assert "status_group" in merged.columns
