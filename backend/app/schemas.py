from typing import Any

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    dataset_id: int
    target_column: str = "churn"
    trigger_reason: str = "manual"


class DriftRequest(BaseModel):
    baseline_dataset_id: int
    current_dataset_id: int
    model_version: str | None = None
    threshold: float = Field(default=0.2, ge=0.01, le=1.0)


class PredictRequest(BaseModel):
    model_version: str | None = None
    features: dict[str, Any]


class MonitorRequest(BaseModel):
    baseline_dataset_id: int
    current_dataset_id: int
    model_version: str | None = None
    drift_threshold: float = Field(default=0.2, ge=0.01, le=1.0)
    degradation_threshold: float = Field(default=0.08, ge=0.0, le=1.0)
    auto_retrain: bool = True


class DemoPredictionRequest(BaseModel):
    dataset_id: int
    model_version: str | None = None
    limit: int = Field(default=25, ge=1, le=250)
