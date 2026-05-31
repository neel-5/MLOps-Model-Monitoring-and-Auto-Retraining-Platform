from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.config import APP_AUTHOR, APP_EMAIL, APP_NAME
from app.database import Base, SessionLocal, engine, get_db
from app.models import ModelVersion
from app.schemas import DemoPredictionRequest, DriftRequest, MonitorRequest, PredictRequest, TrainRequest
from app.services.dataset_service import (
    dataset_to_dict,
    get_dataset_or_404,
    list_datasets,
    register_sample_datasets,
    save_uploaded_dataset,
)
from app.services.drift_service import detect_drift
from app.services.ml_service import (
    evaluate_model_on_dataset,
    get_model_by_version_or_active,
    list_models,
    model_to_dict,
    predict,
    train_model,
)
from app.services.monitoring_service import create_demo_predictions, monitoring_summary, run_monitoring_cycle


def bootstrap_database() -> None:
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        datasets = register_sample_datasets(db)
        if datasets and db.query(ModelVersion).count() == 0:
            baseline = next((dataset for dataset in datasets if "Baseline" in dataset.name), datasets[0])
            train_model(db, baseline.id, baseline.target_column, "initial baseline training")
    finally:
        db.close()


@asynccontextmanager
async def lifespan(_: FastAPI):
    bootstrap_database()
    yield


app = FastAPI(
    title=APP_NAME,
    description="Final-year B.Tech CSE Data Science project by Param Saxena.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    return {
        "name": APP_NAME,
        "owner": APP_AUTHOR,
        "email": APP_EMAIL,
        "docs": "/docs",
        "health": "/api/health",
    }


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": APP_NAME}


@app.get("/api/datasets")
def datasets(db: Session = Depends(get_db)) -> list[dict]:
    register_sample_datasets(db)
    return list_datasets(db)


@app.post("/api/datasets/upload", status_code=201)
def upload_dataset(
    file: UploadFile = File(...),
    target_column: str = Form("churn"),
    display_name: str | None = Form(None),
    db: Session = Depends(get_db),
) -> dict:
    dataset = save_uploaded_dataset(db, file, target_column, display_name)
    return dataset_to_dict(dataset)


@app.get("/api/datasets/{dataset_id}/preview")
def dataset_preview(dataset_id: int, limit: int = 8, db: Session = Depends(get_db)) -> dict:
    dataset = get_dataset_or_404(db, dataset_id)
    df = pd.read_csv(dataset.path).head(max(1, min(limit, 50)))
    return {
        "dataset": dataset_to_dict(dataset),
        "columns": df.columns.tolist(),
        "records": df.where(pd.notnull(df), None).to_dict(orient="records"),
    }


@app.post("/api/train", status_code=201)
def train(request: TrainRequest, db: Session = Depends(get_db)) -> dict:
    model = train_model(db, request.dataset_id, request.target_column, request.trigger_reason)
    return model_to_dict(model)


@app.get("/api/models")
def models(db: Session = Depends(get_db)) -> list[dict]:
    return list_models(db)


@app.get("/api/models/active")
def active_model(db: Session = Depends(get_db)) -> dict:
    model = get_model_by_version_or_active(db)
    return model_to_dict(model)


@app.get("/api/models/{version}")
def model_detail(version: str, db: Session = Depends(get_db)) -> dict:
    model = get_model_by_version_or_active(db, version)
    return model_to_dict(model)


@app.get("/api/models/{version}/evaluate")
def evaluate_model(version: str, dataset_id: int, db: Session = Depends(get_db)) -> dict:
    model = get_model_by_version_or_active(db, version)
    return evaluate_model_on_dataset(db, model, dataset_id)


@app.post("/api/drift")
def drift(request: DriftRequest, db: Session = Depends(get_db)) -> dict:
    return detect_drift(
        db,
        baseline_dataset_id=request.baseline_dataset_id,
        current_dataset_id=request.current_dataset_id,
        threshold=request.threshold,
        model_version=request.model_version,
        persist=True,
    )


@app.post("/api/predict")
def make_prediction(request: PredictRequest, db: Session = Depends(get_db)) -> dict:
    return predict(db, request.features, request.model_version)


@app.post("/api/monitor/run")
def run_monitor(request: MonitorRequest, db: Session = Depends(get_db)) -> dict:
    return run_monitoring_cycle(
        db,
        baseline_dataset_id=request.baseline_dataset_id,
        current_dataset_id=request.current_dataset_id,
        model_version=request.model_version,
        drift_threshold=request.drift_threshold,
        degradation_threshold=request.degradation_threshold,
        auto_retrain=request.auto_retrain,
    )


@app.get("/api/monitoring/summary")
def summary(db: Session = Depends(get_db)) -> dict:
    return monitoring_summary(db)


@app.post("/api/monitoring/demo-predictions")
def demo_predictions(request: DemoPredictionRequest, db: Session = Depends(get_db)) -> dict:
    predictions = create_demo_predictions(
        db,
        dataset_id=request.dataset_id,
        model_version=request.model_version,
        limit=request.limit,
    )
    return {"created": len(predictions), "predictions": predictions}


@app.get("/api/sample-input")
def sample_input(dataset_id: int | None = None, db: Session = Depends(get_db)) -> dict:
    model = get_model_by_version_or_active(db)
    if dataset_id is None:
        dataset_id = model.dataset_id
    dataset = get_dataset_or_404(db, dataset_id)
    df = pd.read_csv(dataset.path)
    if df.empty:
        raise HTTPException(status_code=400, detail="Dataset is empty.")
    missing = [column for column in model.feature_columns if column not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Dataset missing model features: {missing}")
    row = df[model.feature_columns].iloc[0]
    return {
        "dataset_id": dataset.id,
        "model_version": model.version,
        "features": row.where(pd.notnull(row), None).to_dict(),
    }
