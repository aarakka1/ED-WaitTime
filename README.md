# 🏥 Emergency Department Wait Time Prediction

A machine learning project to predict Emergency Department (ED) wait times (represented by the 'Score' metric) using historical hospital data, demonstrating a complete MLOps workflow from Databricks experimentation to model deployment.

## 🚀 Overview

This project focuses on building an accurate regression model to predict the average waiting time/efficiency score for Emergency Departments. The final model is a **Voting Regressor Ensemble**, which was selected as the champion model after rigorous testing and tracking with MLflow.

The entire process, including data loading, preprocessing, model training, and performance logging, was executed within a **Databricks** environment.

### Key Technologies Used

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Data & Compute** | Databricks | Unified data and compute platform for execution. |
| **Model Tracking** | **MLflow** | Experiment tracking, metric logging, and model packaging. |
| **Code & Versioning** | GitHub | Source code control and deployment trigger. |
| **Libraries** | `scikit-learn`, `XGBoost` | Preprocessing and ensemble model training. |

---

## 📊 MLflow Experimentation

The model training was tracked in the Databricks-native MLflow instance. The deployable model is the **VotingRegressorEnsemble**.

### Best Metric Results for VotingRegressorEnsemble

| Metric | Champion Model Result | Description |
| :--- | :--- | :--- |
| **R²** | **[INSERT R² VALUE HERE]** | Coefficient of Determination (closeness of fit). |
| **RMSE** | **[INSERT RMSE VALUE HERE]** | Root Mean Squared Error (prediction error magnitude). |
| **MAE** | **[INSERT MAE VALUE HERE]** | Mean Absolute Error. |

### 🔗 View Full Experiment Results

All training runs, metrics, and model artifacts are logged in the Databricks MLflow UI.

> **[Click here to view the experiment runs in Databricks]** (<PASTE DATABRICKS EXPERIMENT URL HERE>)

---

## 🚀 Deployment (Production Endpoint)

The champion model (`VotingRegressorEnsemble`) was successfully registered in the MLflow Model Registry and deployed as a **real-time API endpoint** for consumption by web applications (like Render).

| Component | Identifier/URL |
| :--- | :--- |
| **Model Name** | `VotingRegressorEnsemble` |
| **Deployment Platform** | Databricks Model Serving |
| **Inference API URL** | **<PASTE DATABRICKS SERVING ENDPOINT URL HERE>** |

To query this endpoint, applications must use a **Databricks Personal Access Token** in the HTTP request header for authentication.
