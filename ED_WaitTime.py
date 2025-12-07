# ============================================================
# FULL FINAL PROJECT CODE — One Single Cell (Final for Deployment)
# ============================================================
# NOTE: The !pip install command has been removed to prevent "SyntaxError: invalid syntax".
# NOTE: MLflow configuration updated to explicitly handle authentication via ENV variables.


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


# ============================================================
# MLflow Configuration (Updated for External Authentication)
# ============================================================
# When running outside Databricks (e.g., on Render), we must explicitly set
# the tracking URI to "databricks" and ensure DATABRICKS_HOST and DATABRICKS_TOKEN
# environment variables are set in the deployment environment.

# If the code runs locally/on Render, it will read DATABRICKS_HOST and DATABRICKS_TOKEN.
mlflow.set_tracking_uri("databricks")

# IMPORTANT: Ensure the username part of the path is correct for the active account!
# This path must exist on your Databricks workspace.
experiment_name = "/Users/arakkalakhila@gmail.com/Final_ED_WaitTime_Project"
mlflow.set_experiment(experiment_name)

print("✅ MLflow tracking: Databricks (Authenticating via DATABRICKS_HOST/TOKEN)")
print("⬆️ Using experiment:", experiment_name)


# ============================================================
# Load Final Dataset
# ============================================================

# NOTE: Ensure FinalDS.xlsx is accessible
file_path = "FinalDS.xlsx"   
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

        # Log all four key regression metrics
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


# ============================================================
# Model Training Runs (Using helper function)
# ============================================================

# Linear Regression
linreg = Pipeline([("prep", preprocessor), ("model", LinearRegression())])
train_and_log("LinearRegression", linreg)

# Random Forest
rf = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
])
train_and_log("RandomForest", rf)

# XGBoost
xgb = Pipeline([
    ("prep", preprocessor),
    ("model", XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, random_state=42))
])
train_and_log("XGBoost", xgb)

# Support Vector Regression (SVR)
svr = Pipeline([("prep", preprocessor), ("model", SVR(kernel="rbf", C=1.0, epsilon=0.1))])
train_and_log("SVR", svr)

# Neural Network Regression (MLPRegressor)
mlp = Pipeline([
    ("prep", preprocessor),
    ("model", MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", max_iter=500, random_state=42))
])
train_and_log("NeuralNetwork", mlp)

# k-NN Regressor
knn = Pipeline([("prep", preprocessor), ("model", KNeighborsRegressor(n_neighbors=5, weights="distance"))])
train_and_log("KNNRegressor", knn)


# ============================================================
# 7) Simple Ensemble (VotingRegressor) - CHAMPION MODEL
# ============================================================

estimators = [
    ('rf', RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
    ('xgb', XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, random_state=42)),
    ('svr', SVR(kernel="rbf", C=1.0, epsilon=0.1))
]

ensemble_pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", VotingRegressor(estimators=estimators, weights=None, n_jobs=-1))
])

train_and_log("VotingRegressorEnsemble", ensemble_pipeline)


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
