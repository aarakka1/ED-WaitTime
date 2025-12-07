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
    missing = [f for f in MODEL_FEATURES if f not in data]
    if missing:
        return False, f"Missing required features: {', '.join(missing)}"

    # basic type checks for numeric features
    numeric_features = ["ZIP Code", "Year", "Month"]
    for f in numeric_features:
        try:
            float(data[f])
        except (ValueError, TypeError):
            return False, f"Invalid value for '{f}': must be numeric."
    return True, ""


def build_dataframe_split(data):
    """
    Build an MLflow 'dataframe_split' style payload
    for a single row of features.
    """
    # keep ordering consistent with MODEL_FEATURES
    row = []
    for f in MODEL_FEATURES:
        if f in ["ZIP Code", "Year", "Month"]:
            row.append(float(data[f]))
        else:
            row.append(str(data[f]))

    payload = {
        "dataframe_split": {
            "columns": MODEL_FEATURES,
            "data": [row],
        }
    }
    return payload


# ============================================================
# Routes
# ============================================================

@app.route("/", methods=["GET"])
def home():
    """
    Render the main HTML page with a form.
    """
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """
    JSON health check (optional).
    """
    return jsonify(
        {
            "status": "ok",
            "message": "ED Wait Time model API is running",
            "serving_url": SERVE_URL,
            "features": MODEL_FEATURES,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict endpoint: accepts JSON, forwards to Databricks MLflow endpoint.
    """
    body = request.get_json()

    if not body:
        return jsonify({"success": False, "error": "No JSON body provided."}), 400

    is_valid, error_msg = validate_input(body)
    if not is_valid:
        return jsonify({"success": False, "error": error_msg}), 400

    payload = build_dataframe_split(body)

    try:
        resp = requests.post(SERVE_URL, headers=HEADERS, data=json.dumps(payload))
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Error calling Databricks endpoint: {str(e)}",
                }
            ),
            500,
        )

    try:
        result = resp.json()
    except ValueError:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Non-JSON response from Databricks: {resp.text}",
                }
            ),
            500,
        )

    # Try to pull a prediction out of typical MLflow response shapes
    if isinstance(result, dict) and "predictions" in result:
        pred_value = result["predictions"][0]
    else:
        pred_value = result

    return jsonify(
        {
            "success": True,
            "prediction": pred_value,
            "raw_response": result,
        }
    )


# ============================================================
# Local dev entrypoint (ignored by gunicorn on Render)
# ============================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
