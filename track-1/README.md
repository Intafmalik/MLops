# Week 17 MLOps — Track A: Telco Customer Churn

An end-to-end MLOps project built using the **IBM Telco Customer Churn** dataset. The project follows a production-style workflow covering data preparation, model training, experiment tracking, model registration, API serving, and data-drift monitoring.

The project uses **uv** for reproducible environments, **MLflow** for experiment tracking and model management, **FastAPI** for serving predictions, and **Evidently AI** for monitoring model/data drift.

---

## 1. Project Overview

The complete workflow is divided into the following stages:

| Step                          | What happens                                                                                                                                   | Code                    |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| **Data Ingestion & Cleaning** | Downloads the Telco dataset, cleans text values, converts `TotalCharges`, encodes `Churn`, and removes the customer ID.                        | `src/data_loader.py`    |
| **Preprocessing**             | Creates a stratified train/test split and applies separate preprocessing for numerical and categorical features using one `ColumnTransformer`. | `src/preprocess.py`     |
| **Model Training**            | Trains Logistic Regression, Random Forest, and Gradient Boosting models and records their results in MLflow.                                   | `src/train.py`          |
| **Model Registration**        | Selects the run with the highest F1 score and registers it as `TelcoChurnClassifier`.                                                          | `src/register_model.py` |
| **API Serving**               | Loads the registered model and exposes prediction and health-check endpoints through FastAPI.                                                  | `src/app.py`            |
| **Drift Monitoring**          | Creates reference/current datasets, introduces synthetic drift, and generates an Evidently drift report.                                       | `src/monitor.py`        |

---

## Project Structure

```text
track-1/
│
├── pyproject.toml              # Project dependencies and supported Python version
├── uv.lock                     # Locked dependency versions for reproducible setup
├── README.md                   # Project documentation
│
├── src/
│   ├── data_loader.py          # Downloads and cleans the Telco churn dataset
│   ├── preprocess.py           # Handles train/test split and feature preprocessing
│   ├── train.py                # Trains the three ML models and logs results to MLflow
│   ├── register_model.py       # Registers the best model and promotes it to Production
│   ├── app.py                  # FastAPI application for model predictions
│   └── monitor.py              # Runs Evidently-based data drift monitoring
│
├── data/                       # Stores the downloaded Telco Customer Churn CSV
├── models/                     # Optional directory for local model fallback files
├── artifacts/                  # Stores generated confusion matrix and ROC curve images
├── reports/                    # Contains the generated Evidently drift report
│
└── mlruns/                     # Local MLflow tracking data and experiment runs
```

The structure above keeps data processing, training, serving, and monitoring code separated, making the project easier to maintain.

---

## 2. Installation with uv

### Why uv is used

This project uses **uv** to keep the Python environment consistent across different machines.

1. **Centralized dependency definition**
   `pyproject.toml` defines the required packages, while `uv.lock` stores the exact resolved versions.

2. **Reproducible installations**
   Running `uv sync` installs dependencies according to the lockfile instead of resolving different versions each time.

3. **Fixed Python version**
   Python 3.11 is used so that the project environment remains consistent.

4. **Simple isolated environment**
   uv automatically creates and manages the `.venv` environment.

### Setup

```bash
# Install uv if it is not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the environment and install locked dependencies
uv sync --python 3.11

# Run project code inside the uv environment
uv run python -m src.train
```

The project is configured for Python `>=3.11,<3.13`, with the lockfile providing reproducible dependency versions.

> MLflow 3.x requires the local file-store setting to be enabled. The project handles this inside the source files, so the local `mlruns/` directory can be used without manually setting the environment variable.

---

## 3. Dataset

The project uses the **IBM Telco Customer Churn** dataset.

* **Dataset:** `Telco-Customer-Churn.csv`
* **Rows:** 7,043
* **Original columns:** 21
* **Columns after cleaning:** 20
* **Target:** `Churn`
* **Churn rate:** approximately 26.5%
* **Customer ID:** removed before training

The dataset contains customer information such as demographics, tenure, billing details, contract information, and subscribed services.

### Data cleaning

`src/data_loader.py` performs the following operations:

* Downloads the dataset when it is not already present.
* Removes unnecessary whitespace.
* Converts `TotalCharges` into a numeric column.
* Handles blank `TotalCharges` values.
* Converts `Churn` from `Yes/No` into `1/0`.
* Removes `customerID`.
* Removes any remaining missing values.

Run the data loader with:

```bash
uv run python -m src.data_loader
```

---

## 4. Model Training

Three classification models are trained:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting

Start training with:

```bash
uv run python -m src.train
```

Each training run is logged to the **`telco-churn`** MLflow experiment.

### MLflow records

For every model, the project stores:

* **Parameters**

  * Model hyperparameters
  * Model family
  * Test size
  * Random state

* **Metrics**

  * Accuracy
  * Precision
  * Recall
  * F1 score
  * ROC-AUC

* **Artifacts**

  * Confusion matrix
  * ROC curve
  * Complete trained Pipeline

The complete Pipeline contains both preprocessing and the classifier, allowing the same preprocessing logic to be used during inference.

---

## 5. MLflow Experiment Tracking and Model Registration

Start the MLflow dashboard using:

```bash
uv run mlflow ui --port 5000
```

Then open:

```text
http://127.0.0.1:5000
```

### Comparing experiments

The `telco-churn` experiment contains:

* Logistic Regression run
* Random Forest run
* Gradient Boosting run
* Drift-monitoring run

The three training runs can be compared using the **F1 score** and other logged metrics.

### Register the selected model

```bash
uv run python -m src.register_model
```

The registration script searches the training runs by F1 score, registers the selected model as:

```text
TelcoChurnClassifier
```

and promotes the model from **Staging to Production**. MLflow 3.x aliases are also supported where traditional stages are deprecated.

### MLflow Screenshots

```text
docs/screenshots/mlflow-runs.png
# MLflow experiment showing the available training and monitoring runs

docs/screenshots/mlflow-run-detail.png
# Detailed MLflow run containing metrics, parameters, and generated artifacts

docs/screenshots/mlflow-registry.png
# Registered TelcoChurnClassifier model and its Production version
```

---

## 6. Model Serving with FastAPI

The FastAPI application loads the registered model using:

```text
models:/TelcoChurnClassifier/Production
```

It also contains fallback options in case the Production model cannot be loaded.

Start the API:

```bash
uv run uvicorn src.app:app --host 0.0.0.0 --port 8000
```

Swagger documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### Available endpoints

| Method | Endpoint   | Purpose                           |
| ------ | ---------- | --------------------------------- |
| `GET`  | `/`        | Returns API and model information |
| `GET`  | `/health`  | Checks service health             |
| `POST` | `/predict` | Generates a churn prediction      |

### Check the service

```bash
curl http://127.0.0.1:8000/
```

Example response:

```json
{
  "message": "Telco Customer Churn API — Track A MLOps",
  "model": "TelcoChurnClassifier",
  "model_source": "models:/TelcoChurnClassifier/Production [sklearn]",
  "model_status": "loaded",
  "endpoints": {
    "docs": "/docs",
    "predict": "POST /predict",
    "health": "GET /health"
  }
}
```

### Predict customer churn

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.1,
    "TotalCharges": 1070.5
  }'
```

Example high-risk response:

```json
{
  "churn_prediction": 1,
  "churn_label": "Yes",
  "churn_probability": 0.6249
}
```

Example low-risk response:

```json
{
  "churn_prediction": 0,
  "churn_label": "No",
  "churn_probability": 0.022
}
```

---

## 7. Data Drift Monitoring with Evidently

Run the monitoring process with:

```bash
uv run python -m src.monitor
```

The monitoring workflow uses a reference dataset and a current dataset to demonstrate how data distribution changes can be detected.

### Monitoring process

1. The cleaned dataset is divided into:

   * **70% reference:** 4,930 rows
   * **30% current:** 2,113 rows

2. Synthetic changes are introduced only into the current dataset:

   * `MonthlyCharges` is increased by 30.
   * A larger portion of `InternetService` is changed to `"Fiber optic"`.

3. Evidently generates a data-drift and classification-quality report.

4. The generated report is saved as:

```text
reports/drift_report.html
```

5. A custom `MonthlyCharges` mean difference is calculated and logged.

6. The report and monitoring metrics are stored in MLflow under the `drift-monitoring` run.

---

## 8. Results and Model Selection

The models were evaluated using a stratified 80/20 test split with `random_state=42`.

| Model                   |   Accuracy |  Precision |     Recall |         F1 |    ROC-AUC |
| ----------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| **Logistic Regression** | **0.8055** |     0.6572 | **0.5588** | **0.6040** | **0.8419** |
| Random Forest           |     0.8027 | **0.6633** |     0.5214 |     0.5838 |     0.8409 |
| Gradient Boosting       |     0.8006 |     0.6598 |     0.5134 |     0.5774 |     0.8405 |

The project uses **F1 score as the main selection metric**. Based on the recorded test results, Logistic Regression achieved the highest F1 score and was registered as `TelcoChurnClassifier` for Production.

---

## 9. Reproduce the Complete Project

Run the following commands in order:

```bash
# 1. Create the reproducible Python environment
uv sync --python 3.11

# 2. Train all three models and log them to MLflow
uv run python -m src.train

# 3. Register the selected model and promote it to Production
uv run python -m src.register_model

# 4. Start the FastAPI prediction service
uv run uvicorn src.app:app --port 8000

# 5. Generate the Evidently drift report
uv run python -m src.monitor

# 6. Open MLflow to inspect experiments and model registry
uv run mlflow ui --port 5000
```

### Fresh environment verification

To test the project from a clean environment:

```bash
rm -rf .venv mlruns
uv sync --python 3.11
uv run python -m src.train
```

This provides a simple way to verify that the project can be reproduced using the locked environment and dependencies.
