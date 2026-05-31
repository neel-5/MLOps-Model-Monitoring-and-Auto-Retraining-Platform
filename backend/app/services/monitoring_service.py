from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from app.models import Dataset, ModelVersion, MonitoringEvent, PredictionLog
from app.services.dataset_service import get_dataset_or_404
from app.services.drift_service import detect_drift, latest_drift_report
from app.services.ml_service import (
    evaluate_model_on_dataset,
    get_model_by_version_or_active,
    model_to_dict,
    predict,
    train_model,
)


def _safe_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def run_monitoring_cycle(
    db: Session,
    baseline_dataset_id: int,
    current_dataset_id: int,
    model_version: str | None = None,
    drift_threshold: float = 0.2,
    degradation_threshold: float = 0.08,
    auto_retrain: bool = True,
) -> dict:
    model = get_model_by_version_or_active(db, model_version)
    drift = detect_drift(
        db,
        baseline_dataset_id=baseline_dataset_id,
        current_dataset_id=current_dataset_id,
        threshold=drift_threshold,
        model_version=model.version,
        persist=True,
    )
    evaluation = evaluate_model_on_dataset(db, model, current_dataset_id)

    baseline_f1 = float(model.metrics.get("f1", 0.0))
    current_f1 = float(evaluation["metrics"].get("f1", 0.0))
    performance_drop = round(max(baseline_f1 - current_f1, 0.0), 4)
    degradation_detected = performance_drop >= degradation_threshold
    retrain_needed = drift["should_retrain"] or degradation_detected

    new_model = None
    action = "observed"
    notes = "Model is healthy."
    if retrain_needed and auto_retrain:
        reasons = []
        if drift["should_retrain"]:
            reasons.append("data drift")
        if degradation_detected:
            reasons.append("performance degradation")
        new_model = train_model(
            db,
            current_dataset_id,
            target_column=model.target_column,
            trigger_reason=f"auto-retraining: {', '.join(reasons)}",
        )
        action = f"auto_retrained_to_{new_model.version}"
        notes = f"Retrained because of {', '.join(reasons)}."
    elif retrain_needed:
        action = "retraining_recommended"
        notes = "Retraining conditions were met, but auto_retrain was disabled."

    event = MonitoringEvent(
        baseline_dataset_id=baseline_dataset_id,
        current_dataset_id=current_dataset_id,
        model_version=model.version,
        baseline_f1=baseline_f1,
        current_f1=current_f1,
        performance_drop=performance_drop,
        drift_score=drift["drift_score"],
        action=action,
        notes=notes,
    )
    db.add(event)
    db.commit()
    db.refresh(event)

    return {
        "event_id": event.id,
        "action": action,
        "notes": notes,
        "baseline_model": model_to_dict(model),
        "new_model": model_to_dict(new_model) if new_model else None,
        "drift": drift,
        "evaluation": evaluation,
        "performance_drop": performance_drop,
        "degradation_detected": degradation_detected,
        "created_at": event.created_at.isoformat() if event.created_at else None,
    }


def create_demo_predictions(
    db: Session,
    dataset_id: int,
    model_version: str | None = None,
    limit: int = 25,
) -> list[dict]:
    dataset = get_dataset_or_404(db, dataset_id)
    model = get_model_by_version_or_active(db, model_version)
    df = pd.read_csv(dataset.path).head(limit)
    feature_columns = model.feature_columns

    predictions = []
    for _, row in df.iterrows():
        features = {column: _safe_value(row.get(column)) for column in feature_columns}
        predictions.append(predict(db, features, model.version))
    return predictions


def monitoring_summary(db: Session) -> dict:
    active_model = db.query(ModelVersion).filter(ModelVersion.is_active).first()
    prediction_logs = db.query(PredictionLog).order_by(PredictionLog.created_at.desc()).limit(250).all()
    recent_predictions = prediction_logs[:15]

    distribution = Counter(log.prediction for log in prediction_logs)
    seven_days_ago = datetime.utcnow() - timedelta(days=6)
    daily_counts: dict[str, int] = defaultdict(int)
    for log in prediction_logs:
        if log.created_at and log.created_at >= seven_days_ago:
            daily_counts[log.created_at.date().isoformat()] += 1

    avg_latency = None
    latencies = [log.latency_ms for log in prediction_logs if log.latency_ms is not None]
    if latencies:
        avg_latency = round(float(sum(latencies) / len(latencies)), 2)

    events = db.query(MonitoringEvent).order_by(MonitoringEvent.created_at.desc()).limit(8).all()

    return {
        "counts": {
            "datasets": db.query(Dataset).count(),
            "model_versions": db.query(ModelVersion).count(),
            "predictions": db.query(PredictionLog).count(),
            "monitoring_events": db.query(MonitoringEvent).count(),
        },
        "active_model": model_to_dict(active_model) if active_model else None,
        "prediction_distribution": dict(distribution),
        "daily_prediction_volume": [
            {"date": day, "count": daily_counts.get(day, 0)}
            for day in sorted(daily_counts.keys())
        ],
        "average_latency_ms": avg_latency,
        "recent_predictions": [
            {
                "id": log.id,
                "model_version": log.model_version.version if log.model_version else None,
                "prediction": log.prediction,
                "probability": log.probability,
                "latency_ms": log.latency_ms,
                "created_at": log.created_at.isoformat() if log.created_at else None,
            }
            for log in recent_predictions
        ],
        "latest_drift": latest_drift_report(db),
        "recent_events": [
            {
                "id": event.id,
                "model_version": event.model_version,
                "baseline_f1": event.baseline_f1,
                "current_f1": event.current_f1,
                "performance_drop": event.performance_drop,
                "drift_score": event.drift_score,
                "action": event.action,
                "notes": event.notes,
                "created_at": event.created_at.isoformat() if event.created_at else None,
            }
            for event in events
        ],
    }
