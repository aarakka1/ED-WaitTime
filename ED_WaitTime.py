import os
import requests
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# =========================
# Env vars (Render)
# =========================
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "").strip()
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "").strip()
MLFLOW_ENDPOINT_URL = os.getenv("MLFLOW_ENDPOINT_URL", "").strip()

if not MLFLOW_ENDPOINT_URL:
    raise RuntimeError("MLFLOW_ENDPOINT_URL must be set as an environment variable.")

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

# =========================
# Measure mapping
# =========================
MEASURE_MAP = {
    "Overall ED wait time":          ("OP_18b", "Average (median) time patients spent in the emergency department before leaving from the visit A lower number of minutes is better"),
    "Wait time (incl. transfers)":   ("OP_18a", "Average (median) time all patients spent in the emergency department before leaving from the visit, including psychiatric/mental health patients and patients who were transferred to another facility. A lower number of minutes is better"),
    "Wait time - psych patients":    ("OP_18c", "Average (median) time patients spent in the emergency department before leaving from the visit- Psychiatric/Mental Health Patients.  A lower number of minutes is better"),
    "Wait time - transfer patients": ("OP_18d", "Average (median) time transfer patients spent in the emergency department before leaving from the visit. A lower number of minutes is better"),
    "ED volume":                     ("EDV",    "Emergency department volume"),
    "Left before being seen":        ("OP_22",  "Left before being seen"),
    "Head CT result time":           ("OP_23",  "Head CT results"),
}

DEFAULT_MEASURE = "Overall ED wait time"

MODEL_FEATURES = [
    "State",
    "County/Parish",
    "Measure ID",
    "Measure Name",
    "ZIP Code",
    "Year",
    "Month",
]

# =========================
# Helpers
# =========================
def validate_input(data):
    required = ["State", "County/Parish", "ZIP Code", "Year", "Month"]
    missing = [f for f in required if f not in data or data[f] in (None, "", [])]
    if missing:
        return False, f"Missing required fields: {', '.join(missing)}"
    for f in ["ZIP Code", "Year", "Month"]:
        try:
            float(data[f])
        except Exception:
            return False, f"Invalid numeric value for: {f}"
    return True, ""

def build_payload(data):
    measure_label = data.get("Measure", DEFAULT_MEASURE)
    measure_id, measure_name = MEASURE_MAP.get(measure_label, MEASURE_MAP[DEFAULT_MEASURE])

    row = [
        str(data["State"]),
        str(data["County/Parish"]),
        measure_id,
        measure_name,
        float(data["ZIP Code"]),
        float(data["Year"]),
        float(data["Month"]),
    ]
    return {"dataframe_split": {"columns": MODEL_FEATURES, "data": [row]}}

# =========================
# Routes
# =========================
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "serve_url": SERVE_URL,
        "features": MODEL_FEATURES,
        "measures": list(MEASURE_MAP.keys()),
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

        payload = build_payload(body)

        try:
            resp = requests.post(
                SERVE_URL,
                headers=HEADERS,
                json=payload,
                timeout=60
            )
        except requests.exceptions.Timeout:
            return jsonify({"success": False, "error": "Databricks request timed out."}), 504
        except requests.exceptions.RequestException as e:
            return jsonify({"success": False, "error": f"Request error: {str(e)}"}), 502

        if resp.status_code != 200:
            return jsonify({
                "success": False,
                "error": "Databricks returned an error",
                "status_code": resp.status_code,
                "databricks_response": resp.text[:1500]
            }), 500

        try:
            result = resp.json()
        except Exception:
            return jsonify({
                "success": False,
                "error": "Databricks returned a non-JSON response",
                "databricks_response": resp.text[:1500]
            }), 500

        if isinstance(result, dict) and "predictions" in result:
            pred = result["predictions"][0]
        else:
            pred = result

        return jsonify({"success": True, "prediction": pred})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
