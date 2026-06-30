import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, render_template, request

from src.config import MODELS_DIR
from src.prediction import predict_from_row

app = Flask(__name__)

with open(MODELS_DIR / "metadata.json", "r", encoding="utf-8") as handle:
    METADATA = json.load(handle)

FEATURES = METADATA.get("feature_list", [])


def build_input_row(form_data: dict) -> dict:
    """Build a one-row feature dictionary from the submitted form values."""
    row: dict[str, object] = {}
    for feature in FEATURES:
        value = form_data.get(feature, "")
        if value == "":
            row[feature] = None
        elif value.lower() in {"true", "false"}:
            row[feature] = value.lower() == "true"
        elif value.isdigit():
            row[feature] = int(value)
        else:
            try:
                row[feature] = float(value)
            except ValueError:
                row[feature] = value
    return row


@app.route("/", methods=["GET"])
def index() -> str:
    """Render the prediction form."""
    return render_template("index.html", prediction=None, error=None)


@app.route("/predict", methods=["POST"])
def predict() -> str:
    """Handle prediction requests and display the result."""
    try:
        row = build_input_row(request.form)
        prediction = predict_from_row(row)
        return render_template("index.html", prediction=prediction, error=None)
    except Exception as exc:
        return render_template("index.html", prediction=None, error=f"Invalid request: {exc}")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
