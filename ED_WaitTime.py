# ============================================================
# FULL FINAL PROJECT CODE — One Single File (Render-ready)
# ============================================================

import os, mlflow, mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans

# This will hold the trained ensemble model for inference
champion_model = None

# ============================================================
# MLflow Configuration
# ============================================================

mlflow.set_tracking_uri("databricks")

experiment_name = "/Users/arakkalakhila@gmail.com/Final_ED_WaitTime_Project"
mlflow.set_experiment(experiment_name)

print("✅ MLflow tracking: Databricks (Authenticating via DATABRICKS_HOST/TOKEN)")
print("⬆️ Using experiment:", experiment_name)

# ============================================================
# Load Final Dataset
# ============================================================

file_path = "FinalDS.xlsx"   # Make sure this file is in the repo root
df = pd.read_excel(file_path)
print("Loaded dataset:", df.shape)

# ============================================================
# Filter to Emergency Department rows
# ============================================================

df_ed = df[df["Condition"] == "Emergency Department"].copy()
df_ed["Score"] = pd.to_numeric(df_ed["Score"], errors="coerce")
df_ed = df_ed.dropna(subset=["Score"])

df_ed["Start Date"] = pd.to_datetime(df_ed["Start Date"], errors="coerce")
df_ed["End Date"]   = pd.to_datetime(df_ed["End Date"], errors="coerce")

df_ed["Year"]  = df_ed["Start Date"].dt.year
df_ed["Month"] = df_ed["Start Date"].dt.month

print("Filtered ED dataset:", df_ed.shape)

# ============================================================
# Select features
# ============================================================

feature_cols_cat = ["State", "County/Parish", "Measure ID", "Measure Name"]
feature_cols_num = ["ZIP Code", "Year", "Month"]

X = df_ed[feature_cols_cat + feature_cols_num].copy()
y = df_ed["Score"].copy()

# ============================================================
# Train/Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# Preprocessing
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_cat),
        ("num", StandardScaler(), feature_cols_num),
    ]
)

# ============================================================
# Helper function — Train + Log to MLflow
# ============================================================

def train_and_log(model_name, model_pipeline):
    with mlflow.start_run(run_name=model_name):
        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_test)

        rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
        r2    = r2_score(y_test, y_pred)
        mae   = mean_absolute_error(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("medae", medae)

        mlflow.sklearn.log_model(
            sk_model=model_pipeline, 
            name=model_name, 
            input_example=X_train.head(5)
        )

        print(f"📌 {model_name}: R²={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")

    return model_pipeline, r2, rmse

# ============================================================
# Model Training Runs
# ============================================================

# 1) Linear Regression
linreg = Pipeline([("prep", preprocessor), ("model", LinearRegression())])
linreg, linreg_r2, linreg_rmse = train_and_log("LinearRegression", linreg)

# 2) Random Forest
rf = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
])
rf, rf_r2, rf_rmse = train_and_log("RandomForest", rf)

# 3) XGBoost
xgb = Pipeline([
    ("prep", preprocessor),
    ("model", XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42
    ))
])
xgb, xgb_r2, xgb_rmse = train_and_log("XGBoost", xgb)

# 4) Support Vector Regression (SVR)
svr = Pipeline([("prep", preprocessor), ("model", SVR(kernel="rbf", C=1.0, epsilon=0.1))])
svr, svr_r2, svr_rmse = train_and_log("SVR", svr)

# 5) Neural Network Regression (MLPRegressor)
mlp = Pipeline([
    ("prep", preprocessor),
    ("model", MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", max_iter=500, random_state=42))
])
mlp, mlp_r2, mlp_rmse = train_and_log("NeuralNetwork", mlp)

# 6) k-NN Regressor
knn = Pipeline([("prep", preprocessor), ("model", KNeighborsRegressor(n_neighbors=5, weights="distance"))])
knn, knn_r2, knn_rmse = train_and_log("KNNRegressor", knn)

# ============================================================
# 7) Simple Ensemble (VotingRegressor) - CHAMPION MODEL
# ============================================================

estimators = [
    ('rf', RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
    ('xgb', XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42
    )),
    ('svr', SVR(kernel="rbf", C=1.0, epsilon=0.1))
]

ensemble_pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", VotingRegressor(estimators=estimators, weights=None, n_jobs=-1))
])

ensemble_pipeline, ensemble_r2, ensemble_rmse = train_and_log("VotingRegressorEnsemble", ensemble_pipeline)

# Set champion model for inference
champion_model = ensemble_pipeline

# ============================================================
# 8) k-Means Clustering (Unsupervised on Features)
# ============================================================

with mlflow.start_run(run_name="KMeansClustering"):
    kmeans_pipeline = Pipeline([
        ("prep", preprocessor),
        ("cluster", KMeans(n_clusters=3, random_state=42, n_init=10))
    ])
    kmeans_pipeline.fit(X_train)
    inertia = kmeans_pipeline.named_steps["cluster"].inertia_
    mlflow.log_metric("inertia", inertia)

    print(f"📌 KMeansClustering: Inertia={inertia:.3f}")

# ============================================================
# Summary
# ============================================================

print("\n🎉 All models logged to Databricks experiment.")
print(f"⬆️ Deployable run: 'VotingRegressorEnsemble'")
print("Ensure DATABRICKS_HOST and DATABRICKS_TOKEN are set in your deployment environment.")

# ============================================================
# Flask API for Render
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON like:
    {
        "State": "VA",
        "County/Parish": "RICHMOND",
        "Measure ID": "ED_1",
        "Measure Name": "Emergency Department Wait Time",
        "ZIP Code": 23219,
        "Year": 2024,
        "Month": 5
    }
    """
    if champion_model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    try:
        # Create a single-row DataFrame with exactly the same columns as training
        input_row = {
            "State": data["State"],
            "County/Parish": data["County/Parish"],
            "Measure ID": data["Measure ID"],
            "Measure Name": data["Measure Name"],
            "ZIP Code": data["ZIP Code"],
            "Year": data["Year"],
            "Month": data["Month"]
        }
        input_df = pd.DataFrame([input_row])

        pred = champion_model.predict(input_df)[0]
        return jsonify({"prediction": float(pred)}), 200

    except KeyError as e:
        return jsonify({"error": f"Missing field in input JSON: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================================
# Local entrypoint (Render will use gunicorn: `gunicorn app:app`)
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
