# Stroke Prediction Project

This project trains and serves a stroke prediction model using the healthcare stroke dataset.

## Project Structure

- `api/` - FastAPI app for real-time predictions.
- `src/` - Reusable training and prediction code.
- `dataset/` - Raw and processed datasets.
- `dataset/archive/` - Unused alternate source data kept for traceability.
- `docs/` - Project and assignment PDFs.
- `models/` - Canonical trained model artifact.
- `Notebooks/` - EDA and modeling notebooks.
- `reports/` - Evaluation reports, metrics, and plots.

## Canonical Model

The production model is:

```text
models/stroke_model_pipeline.joblib
```

It contains the sklearn preprocessing pipeline, selected model, threshold, feature list, and metadata.

## Reproduce Training

```bash
python -m src.train
```

Optional MLflow logging, after installing dependencies:

```bash
python -m src.train --mlflow
```

## Run API

```bash
uvicorn api.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## Prediction Schema

The API accepts the original patient fields. The `bmi_missing` feature is created automatically by `src.predict`, so callers should not send it manually.

## Run Streamlit App

```bash
python -m streamlit run streamlit_app.py
```
