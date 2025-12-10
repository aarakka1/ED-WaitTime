import os
import json
import requests
from flask import Flask, request, jsonify, render_template

# ============================================================
# Flask app - this is what gunicorn will run ("ED_WaitTime:app")
# ============================================================

app = Flask(__name__)

# Environment variables you set in Render
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "").strip()
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "").strip()
MLFLOW_ENDPOINT_URL = os.getenv("MLFLOW_ENDPOINT_URL", "").strip()  # can be full URL or just the path

if not MLFLOW_ENDPOINT_URL:
    raise RuntimeError("MLFLOW_ENDPOINT_URL must be set as an environment variable.")

# Build the final serving URL
if MLFLOW_ENDPOINT_URL.startswith("http"):
    SERVE_URL = MLFLOW_ENDPOINT_URL
else:
    if not DATABRICKS_HOST:
        raise RuntimeError(
            "DATABRICKS_HOST must be set if MLFLOW_ENDPOINT_URL is not a full URL."
        )
    SERVE_URL = DATABRICKS_HOST.rstrip("/") + "/" + MLFLOW_ENDPOINT_URL.lstrip("/")

if not DATABRICKS_TOKEN:
    raise RuntimeError("DATABRICKS_TOKEN must be set as an environment variable.")

HEADERS = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json",
}

# Auto-fill default values for these two features
DEFAULT_MEASURE_ID = "ED_1"
DEFAULT_MEASURE_NAME = "Emergency Department Wait Time"

# The features your model expects
MODEL_FEATURES = [
    "State",
    "County/Parish",
    "Measure ID",
    "Measure Name",
    "ZIP Code",
    "Year",
    "Month",
]


# ============================================================
# Helper: validate and build payload
# ============================================================

def validate_input(data):
    required = ["State", "County/Parish", "ZIP Code", "Year", "Month"]
    missing = [f for f in required if f not in data]
    if missing:
        return False, f"Missing required features: {', '.join(missing)}"

    numeric_features = ["ZIP Code", "Year", "Month"]
    for f in numeric_features:
        try:
            float(data[f])
        except:
            return False, f"Invalid numeric value: {f}"

    return True, ""


def build_dataframe_split(data):
    # Automatically add Measure ID/Name
    data["Measure ID"] = DEFAULT_MEASURE_ID
    data["Measure Name"] = DEFAULT_MEASURE_NAME

    row = []
    for f in MODEL_FEATURES:
        if f in ["ZIP Code", "Year", "Month"]:
            row.append(float(data[f]))
        else:
            row.append(str(data[f]))

    return {
        "dataframe_split": {
            "columns": MODEL_FEATURES,
            "data": [row],
        }
    }


# ============================================================
# Routes
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "features": MODEL_FEATURES})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json()
        if not body:
            return jsonify({"success": False, "error": "No data provided"}), 400

        valid, msg = validate_input(body)
        if not valid:
            return jsonify({"success": False, "error": msg}), 400

        payload = build_dataframe_split(body)
        resp = requests.post(SERVE_URL, headers=HEADERS, data=json.dumps(payload), timeout=15)

        try:
            result = resp.json()
        except:
            return jsonify({"success": False, "error": "Invalid response"}), 500

        pred = result["predictions"][0] if "predictions" in result else result

        return jsonify({"success": True, "prediction": pred})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
