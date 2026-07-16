from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "stroke_model_pipeline.joblib"


class StrokePredictor:
    """Load the trained pipeline and prepare inference records consistently."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self.artifact = joblib.load(self.model_path)
        self.pipeline = self.artifact["pipeline"]
        self.threshold = float(self.artifact["threshold"])
        self.features = list(self.artifact["features"])
        self.model_name = self.artifact.get("model_name", "Logistic Regression")

    @property
    def input_features(self) -> list[str]:
        return [feature for feature in self.features if feature != "bmi_missing"]

    def _records_to_frame(self, records: dict[str, Any] | list[dict[str, Any]]) -> pd.DataFrame:
        if isinstance(records, dict):
            frame = pd.DataFrame([records])
        else:
            frame = pd.DataFrame(records)

        frame = frame.rename(columns={"residence_type": "Residence_type"})

        if "bmi" not in frame.columns:
            frame["bmi"] = pd.NA

        frame["bmi_missing"] = frame["bmi"].isna().astype(int)

        missing = [feature for feature in self.features if feature not in frame.columns]
        if missing:
            raise ValueError(f"Missing required feature(s): {', '.join(missing)}")

        invalid_gender = set(frame["gender"].dropna()) - {"Female", "Male"}
        if invalid_gender:
            values = ", ".join(sorted(str(value) for value in invalid_gender))
            raise ValueError(f"Invalid gender value(s): {values}. Use Female or Male.")

        valid_smoking_status = {"never smoked", "formerly smoked", "smokes"}
        invalid_smoking_status = set(frame["smoking_status"].dropna()) - valid_smoking_status
        if invalid_smoking_status:
            values = ", ".join(sorted(str(value) for value in invalid_smoking_status))
            raise ValueError(
                "Invalid smoking_status value(s): "
                f"{values}. Use never smoked, formerly smoked, or smokes."
            )

        self._normalize_binary_columns(frame, ["hypertension", "heart_disease"])
        frame["age"] = pd.to_numeric(frame["age"], errors="raise").astype(float)

        return frame[self.features]

    def _normalize_binary_columns(self, frame: pd.DataFrame, columns: list[str]) -> None:
        value_map = {
            "No": 0,
            "Yes": 1,
            "no": 0,
            "yes": 1,
            0: 0,
            1: 1,
            0.0: 0,
            1.0: 1,
            False: 0,
            True: 1,
        }

        for column in columns:
            normalized = frame[column].map(value_map)
            if normalized.isna().any():
                invalid = frame.loc[normalized.isna(), column].dropna().unique()
                values = ", ".join(sorted(str(value) for value in invalid))
                raise ValueError(f"Invalid {column} value(s): {values}. Use Yes or No.")
            frame[column] = normalized.astype(int)

    def predict(self, records: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
        frame = self._records_to_frame(records)
        probabilities = self.pipeline.predict_proba(frame)[:, 1]
        predictions = (probabilities >= self.threshold).astype(int)

        return [
            {
                "stroke_probability": float(probability),
                "prediction": int(prediction),
                "threshold": self.threshold,
                "risk_level": self._risk_level(float(probability)),
            }
            for probability, prediction in zip(probabilities, predictions)
        ]

    def predict_one(self, record: dict[str, Any]) -> dict[str, Any]:
        return self.predict(record)[0]

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model_path": str(self.model_path),
            "model_name": self.model_name,
            "threshold": self.threshold,
            "features": self.features,
        }

    def _risk_level(self, probability: float) -> str:
        if probability >= self.threshold:
            return "high"
        if probability >= 0.5:
            return "medium"
        return "low"
