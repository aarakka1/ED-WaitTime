import os
import requests
from flask import Flask, request, jsonify, render_template

# ============================================================
# Flask app - this is what gunicorn will run ("ED_WaitTime:app")
# ============================================================

app = Flask(__name__)

# ============================================================
# Environment variables (set these in Render)
# ============================================================

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "").strip()
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "").strip()
MLFLOW_ENDPOINT_URL = os.getenv("MLFLOW_ENDPOINT_URL", "").strip()  # full URL or path

if not MLFLOW_ENDPOINT_URL:
    raise RuntimeError("MLFLOW_ENDPOINT_URL must be set as an environment variable.")

# Build the final serving URL
if MLFLOW_ENDPOINT_URL.startswith("http"):
    SERVE_URL = MLFLOW_ENDPOINT_URL
else:
    if not DATABRICKS_HOST:
        raise RuntimeError("DATABRICKS_HOST must be set if MLFLOW_ENDPOINT_URL is not a full URL.")
    SERVE_URL = DATABRICKS_HOST.rstrip("/") + "/" + MLFLOW_ENDPOINT_URL.lstrip("/")

if not DATABRICKS_TOKEN:
    raise RuntimeError("DATABRICKS_TOKEN must be set as an environment variable.")

HEADERS = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json",
}

# ============================================================
# Defaults & Model features
# ============================================================

DEFAULT_MEASURE_ID = "ED_1"
DEFAULT_MEASURE_NAME = "Emergency Department Wait Time"

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
# Helpers
# ============================================================

def validate_input(data):
    required = ["State", "County/Parish", "ZIP Code", "Year", "Month"]
    missing = [f for f in required if f not in data or data[f] in (None, "", [])]
    if missing:
        return False, f"Missing required features: {', '.join(missing)}"

    for f in ["ZIP Code", "Year", "Month"]:
        try:
            float(data[f])
        except Exception:
            return False, f"Invalid numeric value: {f}"

    return True, ""


def build_dataframe_split(data):
    # Auto-fill defaults (UI should NOT ask user for these)
    data = dict(data)  # avoid mutating the original request body
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
    # If you want to pass a name into the template:
    # return render_template("index.html", app_name="ED WaitTime")
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "features": MODEL_FEATURES,
        "serve_url_configured": bool(SERVE_URL),
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json(silent=True)
        if not body:
            return jsonify({"success": False, "error": "No data provided"}), 400

        valid, msg = validate_input(body)
        if not valid:
            return jsonify({"success": False, "error": msg}), 400

        payload = build_dataframe_split(body)

        # Call Databricks/MLflow serving endpoint
        try:
            resp = requests.post(
                SERVE_URL,
                headers=HEADERS,
                json=payload,      # cleaner than data=json.dumps(...)
                timeout=45         # more realistic than 15s for serving endpoints
            )
        except requests.exceptions.Timeout:
            return jsonify({"success": False, "error": "Databricks request timed out."}), 504
        except requests.exceptions.RequestException as e:
            return jsonify({"success": False, "error": f"Request error calling Databricks: {str(e)}"}), 502

        # Bubble up Databricks errors (super helpful for debugging)
        if resp.status_code != 200:
            return jsonify({
                "success": False,
                "error": "Databricks returned an error",
                "status_code": resp.status_code,
                "databricks_response": resp.text[:1000],
            }), 500

        # Parse JSON safely
        try:
            result = resp.json()
        except Exception:
            return jsonify({
                "success": False,
                "error": "Databricks returned non-JSON response",
                "databricks_response": resp.text[:1000],
            }), 500

        # Extract prediction
        if isinstance(result, dict) and "predictions" in result and isinstance(result["predictions"], list):
            pred = result["predictions"][0]
        else:
            pred = result

        return jsonify({"success": True, "prediction": pred})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
