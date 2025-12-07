import os
import json
import requests
from flask import Flask, request, jsonify

# ============================================================
# Flask app - this is what gunicorn will run ("ED_WaitTime:app")
# ============================================================

app = Flask(__name__)

# Environment variables you set in Render
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
MLFLOW_ENDPOINT_URL = os.getenv("MLFLOW_ENDPOINT_URL")  # can be full URL or just the path

if not DATABRICKS_HOST or not DATABRICKS_TOKEN or not MLFLOW_ENDPOINT_URL:
    raise RuntimeError(
        "DATABRICKS_HOST, DATABRICKS_TOKEN, and MLFLOW_ENDPOINT_URL "
        "must be set as environment variables."
    )

# Build the final serving URL
if MLFLOW_ENDPOINT_URL.startswith("http"):
    SERVE_URL = MLFLOW_ENDPOINT_URL
else:
    SERVE_URL = DATABRICKS_HOST.rstrip("/") + "/" + MLFLOW_ENDPOINT_URL.lstrip("/")

HEADERS = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json",
}

# ============================================================
# Health check route
# ============================================================

@app.route("/", methods=["GET"])
def healthcheck():
    return jsonify({
        "status": "ok",
        "message": "ED Wait Time model API is running",
        "serving_url": SERVE_URL
    })

# ============================================================
# Predict route
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expected JSON payload:

    {
        "data": [
            {
                "State": "AZ",
                "County/Parish": "Maricopa",
                "Measure ID": "ED_1",
                "Measure Name": "ED Wait Time Score",
                "ZIP Code": 85001,
                "Year": 2024,
                "Month": 10
            }
        ]
    }

    This is forwarded to your Databricks / MLflow serving endpoint as:
    { "inputs": [...] }
    """
    body = request.get_json()

    if not body or "data" not in body:
        return jsonify({"error": "Request JSON must contain a 'data' field (list of rows)."}), 400

    payload = {"inputs": body["data"]}

    resp = requests.post(SERVE_URL, headers=HEADERS, data=json.dumps(payload))
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        return jsonify({
            "error": "Databricks serving endpoint returned an error.",
            "details": str(e),
            "response_text": resp.text
        }), 500

    # Return whatever Databricks responds with (usually predictions)
    return jsonify(resp.json())

# ============================================================
# Local dev entrypoint (ignored by gunicorn)
# ============================================================

if __name__ == "__main__":
    # Useful for testing locally: python ED_WaitTime.py
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
