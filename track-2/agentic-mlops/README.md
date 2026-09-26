# Agentic MLOps — Week 17 (Track B)

This project extends the previous RAG assistant with practical **MLOps practices** for managing environments, experiments, agent behavior, and regression testing.

The project uses:

* **uv** for Python environment and dependency management
* **MLflow** for experiment tracking and agent tracing
* **Evidently AI** for regression testing and evaluation
* **FastAPI** for serving the assistant through an API

The implementation builds on the **Week 15 RAG system** and the **Week 16 single-agent workflow**. Week 17 adds version-controlled prompts, structured execution traces, a 12-query evaluation dataset, MLflow tracking, Evidently testing, and API serving.

---

## Project Structure

```text
agentic-mlops/
│
├── prompts/                         # Versioned prompts used by the agent
│   ├── prompt_v1.txt                # V1 — Basic baseline configuration
│   ├── prompt_v2.txt                # V2 — Adds abstention and keyword retry
│   └── prompt_v3.txt                # V3 — Adds synonyms and improved retrieval
│
├── traces/                          # JSON files containing agent execution traces
│
├── evaluations/                     # Evaluation reports and Evidently input files
│
├── src/                             # Main application source code
│   ├── assistant.py                 # RAG tool using TF-IDF retrieval
│   ├── agent.py                     # Agent loop, prompt versions and tracing
│   ├── tracker.py                   # MLflow parameters, metrics and artifacts
│   ├── evaluate.py                  # Regression evaluation and Evidently tests
│   └── app.py                       # FastAPI endpoints for the assistant
│
├── pyproject.toml                   # Project dependencies and Python configuration
├── uv.lock                          # Locked dependency versions
├── mlflow.db                        # SQLite database used by MLflow
└── README.md                        # Project documentation
```

---

## Setup with uv

### Requirements

* `uv` 0.12+
* Python 3.11

The Python version is kept consistent through:

```bash
uv sync --python 3.11
```

### Installation

```bash
cd agentic-mlops

# Install dependencies and create the virtual environment
uv sync --python 3.11
```

### Optional LLM Configuration

The project can also use live LLM responses through Groq/OpenAI.

Copy the example environment file:

```bash
cp .env.example .env
```

Then configure the required API key inside `.env`.

For example:

```env
GROQ_API_KEY=your_api_key
```

The offline deterministic policies can still be used without an API key.

### Quick Test

```bash
uv run python src/agent.py "What is the refund policy?" v3
```

---

## How uv Is Used

| Task                 | Command                                      |
| -------------------- | -------------------------------------------- |
| Install dependencies | `uv sync --python 3.11`                      |
| Run the agent        | `uv run python src/agent.py "<question>" v3` |
| Run all evaluations  | `uv run python -m src.evaluate --all`        |
| Start the API        | `uv run uvicorn src.app:app --port 8000`     |
| Add a dependency     | `uv add <package>`                           |

The dependencies are defined in `pyproject.toml`, while `uv.lock` stores the resolved dependency versions so that the environment can be reproduced consistently.

---

## Running the Assistant

### Run a Query

Different prompt versions can be tested using:

```bash
uv run python src/agent.py "How do I get my money back?" v1

uv run python src/agent.py "How do I get my money back?" v2

uv run python src/agent.py "How do I get my money back?" v3
```

Each execution generates a trace that is stored inside the `traces/` directory.

### Test Failure Handling

The project also supports failure injection to check whether the agent handles tool errors safely:

```bash
uv run python src/agent.py "What is the refund policy?" v3 --fail
```

### Start the FastAPI Server

```bash
uv run uvicorn src.app:app --port 8000
```

Example API request:

```bash
curl -X POST localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What does the warranty cover?","prompt_version":"v3"}'
```

---

## MLflow Experiment Tracking

MLflow uses a local SQLite database:

```text
mlflow.db
```

The project stores experiment information for each prompt version, including parameters, metrics, and generated artifacts.

### Parameters

The following configuration values are tracked:

* `prompt_version`
* `model_name`
* `temperature`
* `top_k`
* `chunk_size`
* `max_iterations`

### Metrics

The evaluation records:

* Token usage
* Average latency
* Task completion score
* Evidently pass rate
* Correctness
* Information preservation

### Artifacts

MLflow also stores:

* Prompt files
* Agent execution traces
* Evaluation reports
* Evidently input files
* Generated HTML reports when available

### Run Evaluations

```bash
uv run python -m src.evaluate --all
```

### Start MLflow UI

```bash
uv run mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --port 5000
```

Open the MLflow interface and select the:

```text
agentic-mlops
```

experiment.

---

## Experiment Results

The three prompt configurations were tested using the same evaluation process.

| Prompt | Temperature | Top-K | Chunk Size | Max Iterations | Avg. Tokens | Pass Rate |
| ------ | ----------: | ----: | ---------: | -------------: | ----------: | --------: |
| V1     |         0.7 |     3 |        300 |              2 |         948 |     66.7% |
| V2     |         0.3 |     5 |        300 |              3 |        1620 |     75.0% |
| V3     |         0.0 |     5 |        500 |              5 |        2334 |      100% |

---

## Agent Tracing

Every `run_agent()` execution creates a structured trace.

A trace records information about each step of the agent workflow, including:

* Selected tool
* Tool arguments
* Raw tool response
* Agent reasoning
* Next decision
* Number of iterations
* Termination reason
* Token usage
* Latency
* Configuration
* Prompt version

A simplified trace looks like:

```json
{
  "step_number": 1,
  "tool_selected": "rag_search",
  "tool_arguments": {
    "query": "...",
    "top_k": 5
  },
  "raw_tool_result": {
    "ok": true,
    "results": [
      {
        "doc_id": "doc1",
        "score": 10.0,
        "text": "..."
      }
    ]
  },
  "reasoning": "Weak evidence. Retry using keywords and synonyms.",
  "decision": "continue"
}
```

Possible termination reasons include:

```text
answered
clarified
abstained
max_iterations
```

---

## Prompt Version Comparison

### V1 — Basic Baseline

**File:** `prompts/prompt_v1.txt`

V1 represents the initial agent configuration.

Characteristics:

* Performs one raw-question search.
* Attempts to answer even when evidence is weak.
* Does not perform clarification.
* Uses general knowledge when retrieval does not provide enough evidence.

Configuration:

```text
temperature = 0.7
top_k = 3
chunk_size = 300
max_iterations = 2
```

V1 achieved:

```text
8/12 passed — 67%
```

Some failures occurred with paraphrased questions because the exact keywords were not present in the retrieved documents.

---

### V2 — Abstention and Keyword Retry

**File:** `prompts/prompt_v2.txt`

V2 improves the handling of weak evidence and tool failures.

Changes include:

* Keyword extraction
* Retry after weak retrieval
* Abstention on tool failures
* Clarification for vague questions
* JSON-based agent actions

Configuration:

```text
temperature = 0.3
top_k = 5
chunk_size = 300
max_iterations = 3
```

V2 achieved:

```text
9/12 passed — 75%
```

It reduced fabricated responses, but some paraphrased questions still failed because the retry process did not include synonym expansion.

---

### V3 — Improved Production Configuration

**File:** `prompts/prompt_v3.txt`

V3 extends V2 based on the problems identified during testing.

Main improvements:

1. **Synonym expansion**

   The agent uses a synonym mapping to improve retrieval for paraphrased questions.

   Examples include:

   ```text
   money/back → refund/return policy
   forgotten/locked → reset password
   ```

2. **Multiple retrieval iterations**

   For questions requiring information from multiple documents, the agent continues searching until the required information is collected.

3. **Retrieval configuration**

   ```text
   temperature = 0.0
   top_k = 5
   chunk_size = 500
   max_iterations = 5
   ```

V3 achieved:

```text
12/12 passed — 100%
```

---

## Prompt Comparison Summary

| Version | Pass Rate | Main Issues                                             |
| ------- | --------: | ------------------------------------------------------- |
| V1      |       67% | Paraphrases, vague queries and multi-document questions |
| V2      |       75% | Improved safety but still struggled with paraphrases    |
| V3      |      100% | No failures in the 12-query regression set              |

---

## Evidently Regression Testing

The evaluation dataset is defined in:

```text
src/evaluate.py
```

It contains **12 test queries**:

* 8 direct questions
* 2 paraphrased questions with no direct keyword overlap
* 1 vague query
* 1 compositional query requiring information from multiple documents

Run all prompt versions:

```bash
uv run python -m src.evaluate --all
```

Run only V3:

```bash
uv run python -m src.evaluate --version v3
```

### Evaluation Metrics

Each response is compared with its expected answer.

The evaluation checks:

* **Correctness** — Whether the response contains the required information and appropriate document citations.
* **Information Preservation** — Token-level F1 comparison between the response and expected answer.
* **Pass/Fail** — Based on the configured correctness and preservation thresholds.
* **Overall Pass Rate** — Percentage of test cases that pass the evaluation.

The pass condition is:

```text
correctness >= 0.5
AND
preservation >= 0.2
```

For clarification cases, correctness is used as the main evaluation criterion.

---

## Evidently Reports

When the installed Evidently version provides the required `TestSuite` functionality, an HTML report is generated for the corresponding prompt version.

The per-row evaluation data is always stored in:

```text
evaluations/evidently_input_v1.csv
evaluations/evidently_input_v2.csv
evaluations/evidently_input_v3.csv
```

The evaluation can also run without an LLM API key because a deterministic keyword/F1-based judge is available.

---

## Results

| Version | Pass Rate | Correctness | Preservation | Avg. Tokens | Avg. Latency |
| ------- | --------: | ----------: | -----------: | ----------: | -----------: |
| V1      |       67% |        0.69 |         0.59 |         948 |       0.000s |
| V2      |       75% |        0.78 |         0.60 |        1620 |       0.000s |
| V3      |      100% |        0.99 |         0.73 |        2334 |       0.000s |

V3 provides the highest regression-test results in the supplied evaluation set, while V2 provides a lower-token alternative with improved handling compared with V1.

---

## Reproduce the Results

```bash
# Create the environment
uv sync --python 3.11

# Run all prompt evaluations
uv run python -m src.evaluate --all

# Start MLflow
uv run mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --port 5000
```

---

## Week 15 / Week 16 Lineage

This project extends the previous weekly implementations.

### Week 15

```text
src/assistant.py
```

Contains the original RAG functionality based on:

* Chunked documents
* TF-IDF retrieval
* Failure injection

### Week 16

```text
src/agent.py
```

Introduced the basic agent loop:

```text
Decide → Act → Observe
```

along with iteration limits and retrieval behavior.

### Week 17

The current project adds:

* `pyproject.toml`
* `uv.lock`
* Versioned prompts
* Structured agent traces
* Evaluation datasets
* MLflow tracking
* Evidently regression testing
* FastAPI serving
* `src/tracker.py`
* `src/evaluate.py`
* `src/app.py`

Together, these additions turn the earlier RAG and agent prototype into an MLOps-oriented project.
