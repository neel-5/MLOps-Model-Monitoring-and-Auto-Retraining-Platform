from pathlib import Path
import re
import shutil
from datetime import datetime

import pandas as pd
from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import SAMPLE_DATA_DIR, UPLOAD_DIR
from app.models import Dataset


SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def clean_filename(filename: str) -> str:
    name = Path(filename).name.strip().replace(" ", "_")
    return SAFE_FILENAME_RE.sub("_", name)


def inspect_dataset(path: Path, target_column: str = "churn") -> dict:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset file not found: {path.name}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to read CSV: {exc}") from exc

    if df.empty:
        raise HTTPException(status_code=400, detail="Dataset is empty.")

    if target_column not in df.columns:
        if "churn" in df.columns:
            target_column = "churn"
        else:
            target_column = str(df.columns[-1])

    feature_columns = [column for column in df.columns if column != target_column]
    return {
        "target_column": target_column,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "feature_columns": feature_columns,
    }


def dataset_to_dict(dataset: Dataset) -> dict:
    return {
        "id": dataset.id,
        "name": dataset.name,
        "filename": dataset.filename,
        "source": dataset.source,
        "target_column": dataset.target_column,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
        "feature_columns": dataset.feature_columns,
        "created_at": dataset.created_at.isoformat() if dataset.created_at else None,
    }


def register_dataset(
    db: Session,
    path: Path,
    name: str,
    source: str = "sample",
    target_column: str = "churn",
) -> Dataset:
    metadata = inspect_dataset(path, target_column)
    existing = db.query(Dataset).filter(Dataset.name == name).first()
    if existing:
        existing.filename = path.name
        existing.path = str(path)
        existing.source = source
        existing.target_column = metadata["target_column"]
        existing.row_count = metadata["row_count"]
        existing.column_count = metadata["column_count"]
        existing.feature_columns = metadata["feature_columns"]
        db.commit()
        db.refresh(existing)
        return existing

    dataset = Dataset(
        name=name,
        filename=path.name,
        path=str(path),
        source=source,
        target_column=metadata["target_column"],
        row_count=metadata["row_count"],
        column_count=metadata["column_count"],
        feature_columns=metadata["feature_columns"],
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


def register_sample_datasets(db: Session) -> list[Dataset]:
    datasets: list[Dataset] = []
    for csv_path in sorted(SAMPLE_DATA_DIR.glob("*.csv")):
        display_name = csv_path.stem.replace("_", " ").title()
        datasets.append(register_dataset(db, csv_path, display_name, "sample", "churn"))
    return datasets


def list_datasets(db: Session) -> list[dict]:
    datasets = db.query(Dataset).order_by(Dataset.created_at.desc()).all()
    return [dataset_to_dict(dataset) for dataset in datasets]


def get_dataset_or_404(db: Session, dataset_id: int) -> Dataset:
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    if not Path(dataset.path).exists():
        raise HTTPException(status_code=404, detail=f"Dataset file missing: {dataset.filename}")
    return dataset


def save_uploaded_dataset(
    db: Session,
    upload: UploadFile,
    target_column: str = "churn",
    display_name: str | None = None,
) -> Dataset:
    if not upload.filename or not upload.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}_{clean_filename(upload.filename)}"
    destination = UPLOAD_DIR / filename

    with destination.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)

    name = display_name or Path(upload.filename).stem.replace("_", " ").title()
    existing = db.query(Dataset).filter(Dataset.name == name).first()
    if existing:
        name = f"{name} {timestamp}"

    return register_dataset(db, destination, name, "uploaded", target_column)
