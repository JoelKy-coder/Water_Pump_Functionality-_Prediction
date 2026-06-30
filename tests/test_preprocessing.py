import pandas as pd

from src.preprocessing import clean_dataframe, encode_target


def test_clean_dataframe_treats_zero_and_duplicates():
    raw = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "date_recorded": ["2010-01-01", "2010-01-01", "2011-01-01", "2011-01-01"],
            "latitude": [0.0, 0.0, -3.0, -3.0],
            "construction_year": [0, 2000, 2001, 2001],
            "population": [0, 10, 15, 15],
            "public_meeting": ["True", "False", "True", "True"],
            "permit": [1, 0, 1, 1],
            "status_group": ["functional", "functional", "non functional", "non functional"],
        }
    )
    cleaned = clean_dataframe(raw)
    assert cleaned["latitude"].isna().sum() >= 1
    assert cleaned["construction_year"].isna().sum() >= 1
    assert cleaned["population"].isna().sum() >= 1
    assert cleaned.shape[0] == 3


def test_encode_target_returns_encoded_labels():
    df = pd.DataFrame({"status_group": ["functional", "non functional", "functional needs repair"]})
    encoded, encoder = encode_target(df)
    assert set(encoded["status_group"].unique()) == {0, 1, 2}
    assert encoder.classes_.tolist() == ["functional", "functional needs repair", "non functional"]
