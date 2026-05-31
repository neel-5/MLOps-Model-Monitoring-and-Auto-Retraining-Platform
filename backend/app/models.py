from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import relationship

from app.database import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False, index=True)
    filename = Column(String, nullable=False)
    path = Column(String, nullable=False)
    source = Column(String, nullable=False, default="sample")
    target_column = Column(String, nullable=False, default="churn")
    row_count = Column(Integer, nullable=False, default=0)
    column_count = Column(Integer, nullable=False, default=0)
    feature_columns = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)

    model_versions = relationship("ModelVersion", back_populates="dataset")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    version = Column(String, unique=True, nullable=False, index=True)
    model_name = Column(String, nullable=False, default="RandomForestClassifier")
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    artifact_path = Column(String, nullable=False)
    metrics = Column(JSON, nullable=False, default=dict)
    confusion_matrix = Column(JSON, nullable=False, default=list)
    feature_importance = Column(JSON, nullable=False, default=list)
    feature_columns = Column(JSON, nullable=False, default=list)
    target_column = Column(String, nullable=False, default="churn")
    trigger_reason = Column(String, nullable=False, default="manual")
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    dataset = relationship("Dataset", back_populates="model_versions")
    predictions = relationship("PredictionLog", back_populates="model_version")


class DriftReport(Base):
    __tablename__ = "drift_reports"

    id = Column(Integer, primary_key=True, index=True)
    baseline_dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    current_dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    model_version = Column(String, nullable=True)
    drift_score = Column(Float, nullable=False, default=0.0)
    threshold = Column(Float, nullable=False, default=0.2)
    drifted_features = Column(JSON, nullable=False, default=list)
    feature_reports = Column(JSON, nullable=False, default=list)
    should_retrain = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=False)
    input_payload = Column(JSON, nullable=False)
    prediction = Column(String, nullable=False)
    probability = Column(Float, nullable=True)
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    model_version = relationship("ModelVersion", back_populates="predictions")


class MonitoringEvent(Base):
    __tablename__ = "monitoring_events"

    id = Column(Integer, primary_key=True, index=True)
    baseline_dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    current_dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    model_version = Column(String, nullable=False)
    baseline_f1 = Column(Float, nullable=False, default=0.0)
    current_f1 = Column(Float, nullable=False, default=0.0)
    performance_drop = Column(Float, nullable=False, default=0.0)
    drift_score = Column(Float, nullable=False, default=0.0)
    action = Column(String, nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
