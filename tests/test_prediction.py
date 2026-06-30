import pandas as pd

from src.prediction import predict_from_row


def test_predict_from_row_returns_valid_label(monkeypatch):
    class DummyModel:
        def predict(self, frame):
            return [0]

    class DummyEncoder:
        def inverse_transform(self, values):
            return ["functional"]

    monkeypatch.setattr("src.prediction.joblib.load", lambda path: DummyModel() if "best_model" in str(path) else DummyEncoder())
    result = predict_from_row({"amount_tsh": 0.0, "gps_height": 0, "population": 0})
    assert result == "functional"
