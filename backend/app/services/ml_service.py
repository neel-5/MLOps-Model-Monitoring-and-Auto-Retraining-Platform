from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy.orm import Session

from app.config import MODEL_DIR
from app.models import ModelVersion, PredictionLog
from app.services.dataset_service import get_dataset_or_404


IDENTIFIER_COLUMNS = {"id", "customer_id", "user_id", "account_id", "timestamp", "created_at"}


def _clean_feature_name(name: str) -> str:
    return (
        name.replace("numeric__", "")
        .replace("categorical__", "")
        .replace("remainder__", "")
        .replace("_", " ")
    )


def _prepare_xy(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found.")

    feature_columns = [
        column
        for column in df.columns
        if column != target_column and column.lower() not in IDENTIFIER_COLUMNS
    ]
    if not feature_columns:
        raise HTTPException(status_code=400, detail="Dataset does not contain usable feature columns.")

    y = df[target_column]
    if y.nunique() < 2:
        raise HTTPException(status_code=400, detail="Target column must contain at least two classes.")

    return df[feature_columns], y, feature_columns


def _build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    transformers = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)
    classifier = RandomForestClassifier(
        n_estimators=260,
        max_depth=9,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced",
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", classifier)])


def _classification_average(y: pd.Series) -> str:
    labels = set(pd.Series(y).dropna().unique().tolist())
    return "binary" if labels <= {0, 1} and len(labels) == 2 else "weighted"


def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    average = _classification_average(y_true)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average=average, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, average=average, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, average=average, zero_division=0)), 4),
    }


def _feature_importance(pipeline: Pipeline) -> list[dict]:
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    transformed_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_
    pairs = sorted(
        zip(transformed_names, importances, strict=False),
        key=lambda pair: float(pair[1]),
        reverse=True,
    )
    return [
        {"feature": _clean_feature_name(str(name)), "importance": round(float(score), 5)}
        for name, score in pairs[:20]
    ]


def _next_version(db: Session) -> str:
    count = db.query(ModelVersion).count()
    return f"v{count + 1}"


def model_to_dict(model: ModelVersion) -> dict:
    return {
        "id": model.id,
        "version": model.version,
        "model_name": model.model_name,
        "dataset_id": model.dataset_id,
        "artifact_path": model.artifact_path,
        "metrics": model.metrics,
        "confusion_matrix": model.confusion_matrix,
        "feature_importance": model.feature_importance,
        "feature_columns": model.feature_columns,
        "target_column": model.target_column,
        "trigger_reason": model.trigger_reason,
        "is_active": model.is_active,
        "created_at": model.created_at.isoformat() if model.created_at else None,
    }


def train_model(
    db: Session,
    dataset_id: int,
    target_column: str = "churn",
    trigger_reason: str = "manual",
) -> ModelVersion:
    dataset = get_dataset_or_404(db, dataset_id)
    target_column = target_column or dataset.target_column
    df = pd.read_csv(dataset.path)
    X, y, feature_columns = _prepare_xy(df, target_column)

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    pipeline = _build_pipeline(X_train)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = _metrics(y_test, y_pred)
    labels = sorted(pd.Series(y).dropna().unique().tolist())
    matrix = confusion_matrix(y_test, y_pred, labels=labels).tolist()
    importance = _feature_importance(pipeline)

    version = _next_version(db)
    artifact_path = MODEL_DIR / f"{version}_random_forest.joblib"
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_columns": feature_columns,
            "target_column": target_column,
            "labels": labels,
        },
        artifact_path,
    )

    db.query(ModelVersion).update({ModelVersion.is_active: False})
    model_version = ModelVersion(
        version=version,
        model_name="RandomForestClassifier",
        dataset_id=dataset.id,
        artifact_path=str(artifact_path),
        metrics=metrics,
        confusion_matrix={"labels": labels, "matrix": matrix},
        feature_importance=importance,
        feature_columns=feature_columns,
        target_column=target_column,
        trigger_reason=trigger_reason,
        is_active=True,
    )
    db.add(model_version)
    db.commit()
    db.refresh(model_version)
    return model_version


def list_models(db: Session) -> list[dict]:
    models = db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).all()
    return [model_to_dict(model) for model in models]


def get_model_by_version_or_active(db: Session, version: str | None = None) -> ModelVersion:
    query = db.query(ModelVersion)
    model = query.filter(ModelVersion.version == version).first() if version else query.filter(ModelVersion.is_active).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model version not found.")
    if not Path(model.artifact_path).exists():
        raise HTTPException(status_code=404, detail="Model artifact is missing.")
    return model


def load_artifact(model: ModelVersion) -> dict:
    return joblib.load(model.artifact_path)


def evaluate_model_on_dataset(db: Session, model: ModelVersion, dataset_id: int) -> dict:
    dataset = get_dataset_or_404(db, dataset_id)
    artifact = load_artifact(model)
    df = pd.read_csv(dataset.path)
    target_column = artifact["target_column"]
    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in current dataset.")

    feature_columns = artifact["feature_columns"]
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Dataset is missing required columns: {missing}")

    X = df[feature_columns]
    y = df[target_column]
    y_pred = artifact["pipeline"].predict(X)
    metrics = _metrics(y, y_pred)
    labels = artifact.get("labels", sorted(pd.Series(y).dropna().unique().tolist()))
    matrix = confusion_matrix(y, y_pred, labels=labels).tolist()
    return {
        "dataset_id": dataset.id,
        "dataset_name": dataset.name,
        "metrics": metrics,
        "confusion_matrix": {"labels": labels, "matrix": matrix},
    }


def predict(db: Session, features: dict, version: str | None = None) -> dict:
    model = get_model_by_version_or_active(db, version)
    artifact = load_artifact(model)
    expected_features = artifact["feature_columns"]
    payload = {column: features.get(column) for column in expected_features}
    X = pd.DataFrame([payload])

    started = perf_counter()
    prediction = artifact["pipeline"].predict(X)[0]
    probability = None
    if hasattr(artifact["pipeline"], "predict_proba"):
        probabilities = artifact["pipeline"].predict_proba(X)[0]
        probability = round(float(np.max(probabilities)), 4)
    latency_ms = round((perf_counter() - started) * 1000, 2)

    log = PredictionLog(
        model_version_id=model.id,
        input_payload=payload,
        prediction=str(prediction),
        probability=probability,
        latency_ms=latency_ms,
    )
    db.add(log)
    db.commit()
    db.refresh(log)

    return {
        "prediction_id": log.id,
        "model_version": model.version,
        "prediction": str(prediction),
        "probability": probability,
        "latency_ms": latency_ms,
        "input": payload,
    }
