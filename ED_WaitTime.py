import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
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
# Patient type → measure mapping
# =========================
PATIENT_TYPE_MAP = {
    "General":            ("OP_18b", "Average (median) time patients spent in the emergency department before leaving from the visit A lower number of minutes is better"),
    "Transfer":           ("OP_18a", "Average (median) time all patients spent in the emergency department before leaving from the visit, including psychiatric/mental health patients and patients who were transferred to another facility. A lower number of minutes is better"),
    "Psych / Mental Health": ("OP_18c", "Average (median) time patients spent in the emergency department before leaving from the visit- Psychiatric/Mental Health Patients.  A lower number of minutes is better"),
}

# Dashboard metrics (run in parallel after main prediction)
DASHBOARD_MEASURES = {
    "ED Volume":             ("EDV",    "Emergency department volume"),
    "Left Before Being Seen":("OP_22",  "Left before being seen"),
    "Head CT Result Time":   ("OP_23",  "Head CT results"),
    "Transfer Wait Time":    ("OP_18d", "Average (median) time transfer patients spent in the emergency department before leaving from the visit. A lower number of minutes is better"),
}

MODEL_FEATURES = [
    "State", "County/Parish", "Measure ID", "Measure Name",
    "ZIP Code", "Year", "Month",
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

def build_payload(data, measure_id, measure_name):
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

def call_model(data, measure_id, measure_name):
    payload = build_payload(data, measure_id, measure_name)
    resp = requests.post(SERVE_URL, headers=HEADERS, json=payload, timeout=120)
    if resp.status_code != 200:
        return None
    result = resp.json()
    if isinstance(result, dict) and "predictions" in result:
        return round(float(result["predictions"][0]), 1)
    return None

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
        "patient_types": list(PATIENT_TYPE_MAP.keys()),
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

        # Main prediction based on patient type
        patient_type = body.get("PatientType", "General")
        measure_id, measure_name = PATIENT_TYPE_MAP.get(
            patient_type, PATIENT_TYPE_MAP["General"]
        )
        main_pred = call_model(body, measure_id, measure_name)
        if main_pred is None:
            return jsonify({"success": False, "error": "Model call failed for main prediction"}), 500

        # Dashboard metrics in parallel
        dashboard = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(call_model, body, mid, mname): label
                for label, (mid, mname) in DASHBOARD_MEASURES.items()
            }
            for future in as_completed(futures):
                label = futures[future]
                try:
                    dashboard[label] = future.result()
                except Exception:
                    dashboard[label] = None

        return jsonify({
            "success": True,
            "prediction": main_pred,
            "patient_type": patient_type,
            "dashboard": dashboard,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
