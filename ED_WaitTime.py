import os
import time
import requests
import pandas as pd
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
# Load dataset once at startup
# =========================
df_raw = pd.read_excel("FinalDS.xlsx")
df_ed = df_raw[df_raw["Condition"] == "Emergency Department"].copy()
df_ed["Score"] = pd.to_numeric(df_ed["Score"], errors="coerce")
df_ed["Start Date"] = pd.to_datetime(df_ed["Start Date"], errors="coerce")
df_ed["Year"] = df_ed["Start Date"].dt.year
df_ed["Month"] = df_ed["Start Date"].dt.month

# =========================
# In-memory cache
# =========================
prediction_cache = {}
CACHE_TTL = 600  # 10 minutes

def get_cache_key(data, measure_id):
    return f"{data['State']}_{data['County/Parish']}_{data['ZIP Code']}_{data['Year']}_{data['Month']}_{measure_id}"

# =========================
# Patient type → measure mapping
# =========================
PATIENT_TYPE_MAP = {
    "General":               ("OP_18b", "Average (median) time patients spent in the emergency department before leaving from the visit A lower number of minutes is better"),
    "Transfer":              ("OP_18a", "Average (median) time all patients spent in the emergency department before leaving from the visit, including psychiatric/mental health patients and patients who were transferred to another facility. A lower number of minutes is better"),
    "Psych / Mental Health": ("OP_18c", "Average (median) time patients spent in the emergency department before leaving from the visit- Psychiatric/Mental Health Patients.  A lower number of minutes is better"),
}

# Reduced to 2 dashboard calls (down from 4) to cut response time
DASHBOARD_MEASURES = {
    "ED Volume":              ("EDV",   "Emergency department volume"),
    "Left Before Being Seen": ("OP_22", "Left before being seen"),
}

MODEL_FEATURES = [
    "State", "County/Parish", "Measure ID", "Measure Name",
    "ZIP Code", "Year", "Month",
]

# =========================
# Helpers
# =========================
def validate_input(data):
    required = ["State", "ZIP Code", "Year", "Month"]
    missing = [f for f in required if f not in data or data[f] in (None, "", [])]
    if missing:
        return False, f"Missing required fields: {', '.join(missing)}"
    for f in ["ZIP Code", "Year", "Month"]:
        try:
            float(data[f])
        except Exception:
            return False, f"Invalid numeric value for: {f}"
    # Normalize casing
    data["State"] = data["State"].strip().upper()
    # Auto-derive county from ZIP if not provided
    if not data.get("County/Parish"):
        match = df_ed[df_ed["ZIP Code"] == float(data["ZIP Code"])]
        if not match.empty:
            data["County/Parish"] = match.iloc[0]["County/Parish"]
        else:
            data["County/Parish"] = ""
    else:
        data["County/Parish"] = data["County/Parish"].strip().title()
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

def call_model_cached(data, measure_id, measure_name):
    key = get_cache_key(data, measure_id)
    now = time.time()
    if key in prediction_cache:
        cached_time, cached_val = prediction_cache[key]
        if now - cached_time < CACHE_TTL:
            return cached_val
    result = call_model(data, measure_id, measure_name)
    if result is not None:
        prediction_cache[key] = (now, result)
    return result

def get_nearby_hospitals(state, county, patient_type="General"):
    measure_id = PATIENT_TYPE_MAP.get(patient_type, PATIENT_TYPE_MAP["General"])[0]

    nearby = df_ed[
        (df_ed["State"].str.upper() == state.strip().upper()) &
        (df_ed["County/Parish"].str.upper() == county.strip().upper()) &
        (df_ed["Measure ID"] == measure_id)
    ].copy()

    if nearby.empty:
        nearby = df_ed[
            (df_ed["State"].str.upper() == state.strip().upper()) &
            (df_ed["Measure ID"] == measure_id)
        ].copy()

    if nearby.empty:
        return []

    nearby = nearby.sort_values("Year", ascending=False)
    nearby = nearby.drop_duplicates(subset=["Facility Name"])
    nearby = nearby.dropna(subset=["Score"])
    nearby = nearby.sort_values("Score", ascending=True).head(10)

    results = []
    for _, row in nearby.iterrows():
        results.append({
            "name":    row["Facility Name"].title(),
            "address": row["Address"].title() if pd.notna(row["Address"]) else "—",
            "city":    row["City/Town"].title() if pd.notna(row["City/Town"]) else "—",
            "zip":     str(int(row["ZIP Code"])) if pd.notna(row["ZIP Code"]) else "—",
            "phone":   row["Telephone Number"] if pd.notna(row["Telephone Number"]) else "—",
            "wait":    round(float(row["Score"]), 1) if pd.notna(row["Score"]) else None,
            "county":  row["County/Parish"].title() if pd.notna(row["County/Parish"]) else "—",
        })
    return results

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
        "facilities_loaded": int(df_ed["Facility Name"].nunique()),
        "cache_size": len(prediction_cache),
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

        patient_type = body.get("PatientType", "General")
        measure_id, measure_name = PATIENT_TYPE_MAP.get(
            patient_type, PATIENT_TYPE_MAP["General"]
        )

        # 3 total model calls (down from 5) — all cached after first request
        all_calls = {"__main__": (measure_id, measure_name)}
        all_calls.update(DASHBOARD_MEASURES)

        results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(call_model_cached, body, mid, mname): label
                for label, (mid, mname) in all_calls.items()
            }
            for future in as_completed(futures):
                label = futures[future]
                try:
                    results[label] = future.result()
                except Exception:
                    results[label] = None

        main_pred = results.pop("__main__", None)
        if main_pred is None:
            return jsonify({"success": False, "error": "Model call failed for main prediction"}), 500

        nearby = get_nearby_hospitals(
            body["State"], body["County/Parish"], patient_type
        )

        return jsonify({
            "success":      True,
            "prediction":   main_pred,
            "patient_type": patient_type,
            "dashboard":    results,
            "nearby":       nearby,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# =========================
# Databricks warm-up
# =========================
def _warmup():
    """Fire a dummy prediction on startup and every 9 min to keep Databricks hot."""
    warmup_data = {
        "State": "AZ",
        "County/Parish": "Maricopa",
        "ZIP Code": 85001,
        "Year": 2024,
        "Month": 1,
    }
    measure_id, measure_name = PATIENT_TYPE_MAP["General"]
    while True:
        try:
            call_model(warmup_data, measure_id, measure_name)
        except Exception:
            pass
        time.sleep(540)  # 9 minutes

import threading
threading.Thread(target=_warmup, daemon=True).start()

# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
