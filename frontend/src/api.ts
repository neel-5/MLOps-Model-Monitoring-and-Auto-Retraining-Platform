import type { Dataset, DatasetPreview, DriftReport, ModelVersion, MonitoringSummary } from "./types";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: options?.body instanceof FormData ? undefined : { "Content-Type": "application/json" },
    ...options,
  });

  if (!response.ok) {
    const text = await response.text();
    let detail = text;
    try {
      detail = JSON.parse(text).detail ?? text;
    } catch {
      detail = text;
    }
    throw new Error(Array.isArray(detail) ? detail.map((item) => item.msg).join(", ") : detail);
  }

  return response.json() as Promise<T>;
}

export const api = {
  baseUrl: API_BASE,
  datasets: () => request<Dataset[]>("/api/datasets"),
  previewDataset: (datasetId: number) => request<DatasetPreview>(`/api/datasets/${datasetId}/preview?limit=6`),
  uploadDataset: (data: FormData) => request<Dataset>("/api/datasets/upload", { method: "POST", body: data }),
  trainModel: (datasetId: number, targetColumn = "churn") =>
    request<ModelVersion>("/api/train", {
      method: "POST",
      body: JSON.stringify({ dataset_id: datasetId, target_column: targetColumn, trigger_reason: "dashboard training" }),
    }),
  models: () => request<ModelVersion[]>("/api/models"),
  activeModel: () => request<ModelVersion>("/api/models/active"),
  detectDrift: (baselineDatasetId: number, currentDatasetId: number, modelVersion?: string) =>
    request<DriftReport>("/api/drift", {
      method: "POST",
      body: JSON.stringify({
        baseline_dataset_id: baselineDatasetId,
        current_dataset_id: currentDatasetId,
        model_version: modelVersion,
        threshold: 0.2,
      }),
    }),
  runMonitor: (baselineDatasetId: number, currentDatasetId: number, modelVersion?: string) =>
    request<Record<string, unknown>>("/api/monitor/run", {
      method: "POST",
      body: JSON.stringify({
        baseline_dataset_id: baselineDatasetId,
        current_dataset_id: currentDatasetId,
        model_version: modelVersion,
        drift_threshold: 0.2,
        degradation_threshold: 0.08,
        auto_retrain: true,
      }),
    }),
  summary: () => request<MonitoringSummary>("/api/monitoring/summary"),
  demoPredictions: (datasetId: number, modelVersion?: string) =>
    request<{ created: number }>("/api/monitoring/demo-predictions", {
      method: "POST",
      body: JSON.stringify({ dataset_id: datasetId, model_version: modelVersion, limit: 40 }),
    }),
  sampleInput: (datasetId?: number) =>
    request<{ features: Record<string, string | number | null>; model_version: string }>(
      `/api/sample-input${datasetId ? `?dataset_id=${datasetId}` : ""}`,
    ),
  predict: (features: Record<string, string | number | null>, modelVersion?: string) =>
    request<Record<string, unknown>>("/api/predict", {
      method: "POST",
      body: JSON.stringify({ model_version: modelVersion, features }),
    }),
};
