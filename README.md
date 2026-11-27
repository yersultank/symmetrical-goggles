# Wine Classifier: FastAPI + Docker + MLflow + Streamlit

A wine classification project (scikit-learn Wine dataset, RandomForest model).
The model is served through FastAPI, packaged with Docker, experiments are tracked with MLflow, and the user interface is built with Streamlit.

## Project Structure

```
├── train.py              # model training + MLflow logging
├── main.py               # FastAPI application
├── app.py                # Streamlit frontend
├── model.joblib          # saved model
├── requirements.txt      # API dependencies (used by Docker)
├── requirements-ui.txt   # MLflow and Streamlit dependencies
├── Dockerfile
└── screenshots/
```

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-ui.txt
```

## Practical Task 6: API and Docker

Train the model:

```bash
python train.py
```

Run the API locally:

```bash
uvicorn main:app --reload
```

Endpoints:

- `GET /` returns `{"message": "ML API is running"}`
- `POST /predict` accepts 13 wine features and returns the predicted class
- `/docs` interactive Swagger documentation

Example request to `/predict`:

```json
{
  "alcohol": 13.2, "malic_acid": 2.77, "ash": 2.51,
  "alcalinity_of_ash": 18.5, "magnesium": 98.0, "total_phenols": 1.96,
  "flavanoids": 0.9, "nonflavanoid_phenols": 0.5,
  "proanthocyanins": 1.14, "color_intensity": 5.0, "hue": 0.95,
  "od280_od315_of_diluted_wines": 2.0, "proline": 700.0
}
```

Example response:

```json
{"prediction": 2, "class_name": "class_2"}
```

Docker:

```bash
docker build -t wine-api .
docker run -p 8000:8000 wine-api
```

## SIS-3: MLflow and Streamlit

The following was added to the project:

- **MLflow Experiment Tracking**: experiment `wine-classification`. Each run logs hyperparameters (`n_estimators`, `max_depth`, `random_state`), metrics (`accuracy`, `f1_macro`), and artifacts (`model.joblib` and the model in MLflow format).
- **MLflow Model Registry**: the model is registered as `WineClassifier`, and every run of `train.py` creates a new version.
- **Streamlit frontend**: a web form for entering features that sends a request to the FastAPI service and displays the result.

### How to Run

Run each command in a separate terminal (with the `venv` activated):

| Component | Command | URL |
|---|---|---|
| FastAPI | `uvicorn main:app --reload` | http://127.0.0.1:8000/docs |
| Streamlit | `streamlit run app.py` | http://localhost:8501 |
| MLflow UI | `mlflow ui --backend-store-uri sqlite:///mlflow.db` | http://127.0.0.1:5000 |

Experiments are launched with `python train.py` (change the model parameters in the `params` dictionary inside `train.py`).

### Screenshots

MLflow: list of experiment runs

![MLflow runs](screenshots/mlflow_runs.png)

MLflow: run details (parameters, metrics, artifacts)

![MLflow run details](screenshots/mlflow_run_details.png)

MLflow Model Registry: `WineClassifier` model

![Model registry](screenshots/mlflow_registry.png)

Streamlit interface

![Streamlit](screenshots/streamlit.png)