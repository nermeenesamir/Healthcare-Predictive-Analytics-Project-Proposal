from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import StrokePredictor


app = FastAPI(
    title="Stroke Prediction API",
    version="1.0.0",
    description="Real-time API for the trained stroke prediction pipeline.",
)

predictor = StrokePredictor()


class PatientInput(BaseModel):
    gender: Literal["Female", "Male"]
    age: int = Field(..., ge=0, le=120)
    hypertension: Literal["No", "Yes"]
    heart_disease: Literal["No", "Yes"]
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float = Field(..., ge=0)
    bmi: float | None = Field(default=None, ge=0)
    smoking_status: Literal["never smoked", "formerly smoked", "smokes"]

    def to_record(self) -> dict[str, Any]:
        if hasattr(self, "model_dump"):
            data = self.model_dump()
        else:
            data = self.dict()

        data["age"] = float(data["age"])
        data["hypertension"] = 1 if data["hypertension"] == "Yes" else 0
        data["heart_disease"] = 1 if data["heart_disease"] == "Yes" else 0
        return data


class BatchPredictionRequest(BaseModel):
    records: list[PatientInput]


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Stroke prediction API is running."}


@app.get("/health")
def health() -> dict[str, Any]:
    return predictor.health()


@app.post("/predict")
def predict(patient: PatientInput) -> dict[str, Any]:
    try:
        return predictor.predict_one(patient.to_record())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/predict-batch")
def predict_batch(payload: BatchPredictionRequest) -> dict[str, Any]:
    try:
        predictions = predictor.predict([record.to_record() for record in payload.records])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {"count": len(predictions), "predictions": predictions}
