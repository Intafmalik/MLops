# Week 17: MLOps — Track A (Data Science) + Track B (Agentic AI)

This repository contains the **Week 17 MLOps problem set**, covering practical MLOps concepts such as environment management, experiment tracking, model serving, and monitoring.

The work is divided into two tracks:

* **Track A:** MLOps workflow for a classical ML model using the Telco Customer Churn dataset.
* **Track B:** MLOps practices applied to the W15/W16 RAG and agentic AI assistant.

Each track maintains its own environment configuration and `uv.lock` file for reproducibility.

---

## Tools and Technologies

| Tool             | Track A — Data Science (`track-1/`)              | Track B — Agentic AI (`track-2/`)                              |
| ---------------- | ------------------------------------------------ | -------------------------------------------------------------- |
| **uv**           | Manages a reproducible ML environment            | Manages the assistant's environment                            |
| **MLflow**       | Tracks experiments and manages the trained model | Tracks prompt/configuration experiments and evaluation results |
| **Evidently AI** | Detects data and target drift                    | Performs regression and LLM evaluation                         |
| **Airflow**      | Not implemented                                  | Not implemented                                                |

---

## Repository Structure

```text
mlops/
│
├── README.md                         # Main documentation for the complete project
├── .gitignore                        # Files/folders excluded from Git
│
├── track-1/                          # Track A — Telco Customer Churn MLOps
│   │
│   ├── pyproject.toml                # Python project configuration and dependencies
│   ├── uv.lock                       # Locked dependency versions for reproducibility
│   │
│   ├── src/
│   │   ├── data_loader.py            # Downloads and prepares the Telco dataset
│   │   ├── preprocess.py             # Splits data and applies preprocessing
│   │   ├── train.py                  # Trains the three ML model configurations
│   │   ├── register_model.py         # Registers the selected model in MLflow
│   │   ├── app.py                    # FastAPI application for model prediction
│   │   └── monitor.py                # Runs data-drift monitoring with Evidently
│   │
│   ├── data/                         # Dataset downloaded during execution
│   ├── artifacts/                    # Generated evaluation plots and model outputs
│   ├── reports/                      # Generated HTML monitoring reports
│   ├── models/                       # Optional locally saved model files
│   └── mlruns/                       # Local MLflow experiment and model data
│
└── track-2/                          # Track B — Agentic AI MLOps
    │
    ├── w16/                          # Previous W16 RAG/agent baseline
    │   │                              # Contains RAG, agent, evaluation and result files
    │
    └── agentic-mlops/                # Main Track B MLOps implementation
        │
        ├── pyproject.toml            # Project dependencies and Python configuration
        ├── uv.lock                   # Locked dependency versions
        │
        ├── prompts/                  # Versioned assistant prompts
        │   ├── prompt_v1.txt         # Initial prompt configuration
        │   ├── prompt_v2.txt         # Improved prompt configuration
        │   └── prompt_v3.txt         # More robust prompt configuration
        │
        ├── src/                      # Main application source code
        │   ├── assistant.py          # Assistant functionality
        │   ├── agent.py              # Agent workflow and trace generation
        │   ├── tracker.py            # MLflow tracking
        │   ├── evaluate.py           # Evaluation and regression testing
        │   └── app.py                # FastAPI application
        │
        ├── traces/                   # Saved agent execution traces
        ├── evaluations/              # Evaluation results and audit files
        └── mlflow.db                 # SQLite database used by MLflow
```

---

## Track A — Data Science MLOps

**Directory:** `track-1/`

### Objective

Apply an end-to-end MLOps workflow to a customer churn prediction problem.

The workflow follows:

**Data → Preprocessing → Training → Tracking → Model Registry → Serving → Monitoring**

### 1. Environment Management

The project uses `uv` to create a reproducible Python environment.

```bash
cd track-1
uv sync --python 3.11
```

Run the training pipeline with:

```bash
uv run python -m src.train
```

The `pyproject.toml` file contains the project dependencies, while `uv.lock` keeps the dependency versions consistent.

### 2. Experiment Tracking

Three different machine-learning models are trained and tracked using MLflow:

| Model               |   Accuracy |  Precision |     Recall |         F1 |    ROC-AUC |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression | **0.8055** |     0.6572 | **0.5588** | **0.6040** | **0.8419** |
| Random Forest       |     0.8027 | **0.6633** |     0.5214 |     0.5838 |     0.8409 |
| Gradient Boosting   |     0.8006 |     0.6598 |     0.5134 |     0.5774 |     0.8405 |

MLflow records:

* Model parameters
* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrix
* ROC curve
* Trained model pipeline

The selected model is registered as:

```text
TelcoChurnClassifier
```

Start MLflow UI:

```bash
uv run mlflow ui --port 5000
```

### 3. Model Serving

The trained model is served through FastAPI.

```bash
uv run uvicorn src.app:app --port 8000
```

Available endpoints:

| Method | Endpoint   | Description                              |
| ------ | ---------- | ---------------------------------------- |
| GET    | `/`        | Displays service information             |
| GET    | `/health`  | Checks API status                        |
| POST   | `/predict` | Returns churn prediction and probability |

### 4. Monitoring

Evidently AI is used to identify changes between reference data and current data.

The monitoring process:

* Creates reference and current datasets.
* Introduces controlled data changes.
* Detects the resulting drift.
* Generates an HTML report.
* Logs monitoring results to MLflow.

Generated report:

```text
track-1/reports/drift_report.html
```

---

## Track B — Agentic AI MLOps

**Directory:** `track-2/agentic-mlops/`

### Objective

Apply MLOps practices to the W15/W16 RAG assistant and agentic workflow.

Instead of focusing mainly on traditional model training, this track evaluates:

* Prompt versions
* Retrieval configuration
* Agent behavior
* Execution traces
* Response quality
* Regression performance

### 1. Environment Management

Create the environment with:

```bash
cd track-2/agentic-mlops
uv sync --python 3.11
```

Dependencies are defined in `pyproject.toml` and locked in `uv.lock`.

### 2. Experiment Tracking

Three prompt versions are tested:

* **V1:** Initial configuration
* **V2:** Adds abstention and retry behavior
* **V3:** Adds synonym handling and improved multi-step processing

| Version |        Pass Rate | Correctness | Preservation | Avg. Tokens |
| ------- | ---------------: | ----------: | -----------: | ----------: |
| V1      |       8/12 (67%) |        0.69 |         0.59 |         948 |
| V2      |       9/12 (75%) |        0.78 |         0.60 |        1620 |
| V3      | **12/12 (100%)** |        0.99 |         0.73 |        2334 |

Execution traces are stored in:

```text
traces/
```

MLflow uses:

```text
mlflow.db
```

Start MLflow:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

### 3. Monitoring and Regression Testing

A fixed set of representative queries is used as the evaluation set.

The evaluation checks:

* Response correctness
* Information preservation
* Document citation
* Regression between prompt versions

Evaluation files are stored in:

```text
evaluations/
```

The results can be used to compare different prompt and agent configurations.

### 4. Serving

Start the Track B API:

```bash
uv run uvicorn src.app:app --port 8000
```

Main endpoints:

| Method | Endpoint                 | Description                    |
| ------ | ------------------------ | ------------------------------ |
| POST   | `/query`                 | Sends a query to the assistant |
| POST   | `/evaluate`              | Runs evaluation                |
| GET    | `/prompts`               | Lists prompt versions          |
| GET    | `/evaluations/{version}` | Gets evaluation results        |
| GET    | `/health`                | Checks API status              |

---

## Documentation Cross-Reference

| MLOps Requirement             | Implementation                    |
| ----------------------------- | --------------------------------- |
| Environment & Reproducibility | `uv`, `pyproject.toml`, `uv.lock` |
| Experiment Tracking           | MLflow                            |
| Model Registry                | MLflow Model Registry             |
| Data Monitoring               | Evidently AI                      |
| Regression Testing            | Evidently + evaluation scripts    |
| API Serving                   | FastAPI                           |
| Orchestration                 | Airflow — not implemented         |

---

## Submission Checklist

* [x] GitHub repository with both tracks
* [x] `pyproject.toml` and `uv.lock`
* [x] MLflow experiment tracking
* [x] Model registration for Track A
* [x] Evidently monitoring
* [x] Evaluation and regression testing for Track B
* [ ] Airflow DAGs — not implemented

---

## Reproduce Everything

### Track A

```bash
cd track-1

uv sync --python 3.11

uv run python -m src.train

uv run python -m src.register_model

uv run python -m src.monitor

uv run uvicorn src.app:app --port 8000
```

### Track B

```bash
cd track-2/agentic-mlops

uv sync --python 3.11

uv run python -m src.evaluate --all

uv run uvicorn src.app:app --port 8000
```

### W16 Baseline

```bash
cd track-2/w16

python3 eval.py
```

---

## Project Documentation

Detailed documentation for each part is available in:

* `track-1/README.md`
* `track-2/agentic-mlops/README.md`
* `track-2/w16/README.md`
