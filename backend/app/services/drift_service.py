import math

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models import DriftReport
from app.services.dataset_service import get_dataset_or_404
from app.services.ml_service import IDENTIFIER_COLUMNS


def _numeric_psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()
    if expected.empty or actual.empty:
        return 0.0

    combined_min = min(float(expected.min()), float(actual.min()))
    combined_max = max(float(expected.max()), float(actual.max()))
    if math.isclose(combined_min, combined_max):
        return 0.0

    breakpoints = np.nanpercentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        breakpoints = np.linspace(combined_min, combined_max, buckets + 1)

    breakpoints[0] = combined_min - 1e-6
    breakpoints[-1] = combined_max + 1e-6
    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_pct = np.maximum(expected_counts / max(len(expected), 1), 1e-6)
    actual_pct = np.maximum(actual_counts / max(len(actual), 1), 1e-6)
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return round(float(max(psi, 0.0)), 4)


def _categorical_distance(expected: pd.Series, actual: pd.Series) -> float:
    expected_dist = expected.fillna("__missing__").astype(str).value_counts(normalize=True)
    actual_dist = actual.fillna("__missing__").astype(str).value_counts(normalize=True)
    categories = sorted(set(expected_dist.index) | set(actual_dist.index))
    distance = 0.0
    for category in categories:
        distance += abs(float(actual_dist.get(category, 0.0)) - float(expected_dist.get(category, 0.0)))
    return round(float(distance / 2), 4)


def _feature_columns(df: pd.DataFrame, target_column: str) -> list[str]:
    return [
        column
        for column in df.columns
        if column != target_column and column.lower() not in IDENTIFIER_COLUMNS
    ]


def detect_drift(
    db: Session,
    baseline_dataset_id: int,
    current_dataset_id: int,
    threshold: float = 0.2,
    model_version: str | None = None,
    persist: bool = True,
) -> dict:
    baseline = get_dataset_or_404(db, baseline_dataset_id)
    current = get_dataset_or_404(db, current_dataset_id)
    baseline_df = pd.read_csv(baseline.path)
    current_df = pd.read_csv(current.path)

    baseline_features = _feature_columns(baseline_df, baseline.target_column)
    current_features = _feature_columns(current_df, current.target_column)
    common_features = [column for column in baseline_features if column in current_features]
    if not common_features:
        raise HTTPException(status_code=400, detail="No common feature columns available for drift detection.")

    reports = []
    for feature in common_features:
        if pd.api.types.is_numeric_dtype(baseline_df[feature]) and pd.api.types.is_numeric_dtype(current_df[feature]):
            score = _numeric_psi(baseline_df[feature], current_df[feature])
            method = "population_stability_index"
        else:
            score = _categorical_distance(baseline_df[feature], current_df[feature])
            method = "categorical_distribution_distance"

        reports.append(
            {
                "feature": feature,
                "score": score,
                "method": method,
                "status": "drifted" if score >= threshold else "stable",
            }
        )

    reports.sort(key=lambda item: item["score"], reverse=True)
    drifted = [item["feature"] for item in reports if item["score"] >= threshold]
    drift_score = round(float(max(item["score"] for item in reports)), 4)
    should_retrain = bool(drifted)

    response = {
        "baseline_dataset_id": baseline.id,
        "baseline_dataset_name": baseline.name,
        "current_dataset_id": current.id,
        "current_dataset_name": current.name,
        "model_version": model_version,
        "threshold": threshold,
        "drift_score": drift_score,
        "drifted_features": drifted,
        "feature_reports": reports,
        "should_retrain": should_retrain,
    }

    if persist:
        report = DriftReport(
            baseline_dataset_id=baseline.id,
            current_dataset_id=current.id,
            model_version=model_version,
            drift_score=drift_score,
            threshold=threshold,
            drifted_features=drifted,
            feature_reports=reports,
            should_retrain=should_retrain,
        )
        db.add(report)
        db.commit()
        db.refresh(report)
        response["id"] = report.id
        response["created_at"] = report.created_at.isoformat() if report.created_at else None

    return response


def latest_drift_report(db: Session) -> dict | None:
    report = db.query(DriftReport).order_by(DriftReport.created_at.desc()).first()
    if not report:
        return None
    return {
        "id": report.id,
        "baseline_dataset_id": report.baseline_dataset_id,
        "current_dataset_id": report.current_dataset_id,
        "model_version": report.model_version,
        "drift_score": report.drift_score,
        "threshold": report.threshold,
        "drifted_features": report.drifted_features,
        "feature_reports": report.feature_reports,
        "should_retrain": report.should_retrain,
        "created_at": report.created_at.isoformat() if report.created_at else None,
    }
