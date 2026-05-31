export type Dataset = {
  id: number;
  name: string;
  filename: string;
  source: string;
  target_column: string;
  row_count: number;
  column_count: number;
  feature_columns: string[];
  created_at: string | null;
};

export type Metrics = {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
};

export type FeatureImportance = {
  feature: string;
  importance: number;
};

export type ConfusionMatrix = {
  labels: Array<string | number>;
  matrix: number[][];
};

export type ModelVersion = {
  id: number;
  version: string;
  model_name: string;
  dataset_id: number;
  metrics: Metrics;
  confusion_matrix: ConfusionMatrix;
  feature_importance: FeatureImportance[];
  feature_columns: string[];
  target_column: string;
  trigger_reason: string;
  is_active: boolean;
  created_at: string | null;
};

export type DriftFeatureReport = {
  feature: string;
  score: number;
  method: string;
  status: "stable" | "drifted";
};

export type DriftReport = {
  id?: number;
  baseline_dataset_id: number;
  baseline_dataset_name?: string;
  current_dataset_id: number;
  current_dataset_name?: string;
  model_version?: string | null;
  threshold: number;
  drift_score: number;
  drifted_features: string[];
  feature_reports: DriftFeatureReport[];
  should_retrain: boolean;
  created_at?: string | null;
};

export type PredictionLog = {
  id: number;
  model_version: string | null;
  prediction: string;
  probability: number | null;
  latency_ms: number | null;
  created_at: string | null;
};

export type MonitoringEvent = {
  id: number;
  model_version: string;
  baseline_f1: number;
  current_f1: number;
  performance_drop: number;
  drift_score: number;
  action: string;
  notes: string | null;
  created_at: string | null;
};

export type MonitoringSummary = {
  counts: {
    datasets: number;
    model_versions: number;
    predictions: number;
    monitoring_events: number;
  };
  active_model: ModelVersion | null;
  prediction_distribution: Record<string, number>;
  daily_prediction_volume: Array<{ date: string; count: number }>;
  average_latency_ms: number | null;
  recent_predictions: PredictionLog[];
  latest_drift: DriftReport | null;
  recent_events: MonitoringEvent[];
};

export type DatasetPreview = {
  dataset: Dataset;
  columns: string[];
  records: Array<Record<string, string | number | null>>;
};
