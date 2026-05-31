import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Database,
  GitBranch,
  Play,
  RefreshCcw,
  Rocket,
  ShieldCheck,
  Upload,
  Wand2,
} from "lucide-react";
import { api } from "./api";
import { Bars } from "./components/Bars";
import { ConfusionMatrix } from "./components/ConfusionMatrix";
import { MetricCard } from "./components/MetricCard";
import { StatusBadge } from "./components/StatusBadge";
import type { Dataset, DatasetPreview, DriftReport, ModelVersion, MonitoringSummary } from "./types";

function pct(value?: number) {
  return `${Math.round((value ?? 0) * 100)}%`;
}

function formatDate(value?: string | null) {
  if (!value) return "Not available";
  return new Intl.DateTimeFormat("en-IN", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

export default function App() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [models, setModels] = useState<ModelVersion[]>([]);
  const [summary, setSummary] = useState<MonitoringSummary | null>(null);
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [selectedBaseline, setSelectedBaseline] = useState<number | null>(null);
  const [selectedCurrent, setSelectedCurrent] = useState<number | null>(null);
  const [driftReport, setDriftReport] = useState<DriftReport | null>(null);
  const [predictionResult, setPredictionResult] = useState<Record<string, unknown> | null>(null);
  const [message, setMessage] = useState("Connecting to FastAPI backend...");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const activeModel = summary?.active_model ?? models.find((model) => model.is_active) ?? null;
  const latestDrift = driftReport ?? summary?.latest_drift ?? null;

  const selectedCurrentDataset = useMemo(
    () => datasets.find((dataset) => dataset.id === selectedCurrent) ?? null,
    [datasets, selectedCurrent],
  );

  async function loadAll() {
    setError(null);
    try {
      const [datasetData, modelData, summaryData] = await Promise.all([api.datasets(), api.models(), api.summary()]);
      setDatasets(datasetData);
      setModels(modelData);
      setSummary(summaryData);
      setMessage(`Connected to ${api.baseUrl}`);

      if (!selectedBaseline && datasetData.length) {
        setSelectedBaseline((datasetData.find((dataset) => dataset.name.includes("Baseline")) ?? datasetData[0]).id);
      }
      if (!selectedCurrent && datasetData.length) {
        setSelectedCurrent(
          (datasetData.find((dataset) => dataset.name.includes("Drifted")) ??
            datasetData.find((dataset) => dataset.name.includes("Current")) ??
            datasetData[0]).id,
        );
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Backend is not reachable.");
      setMessage("Backend connection pending");
    }
  }

  useEffect(() => {
    loadAll();
    const timer = window.setInterval(loadAll, 12000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!selectedCurrent) return;
    api
      .previewDataset(selectedCurrent)
      .then(setPreview)
      .catch((err) => setError(err instanceof Error ? err.message : "Could not load dataset preview."));
  }, [selectedCurrent]);

  async function runAction(label: string, action: () => Promise<void>) {
    setBusy(label);
    setError(null);
    try {
      await action();
      await loadAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Action failed.");
    } finally {
      setBusy(null);
    }
  }

  function requireSelection() {
    if (!selectedBaseline || !selectedCurrent) {
      throw new Error("Select baseline and current datasets first.");
    }
  }

  async function handleUpload(file?: File | null) {
    if (!file) return;
    await runAction("upload", async () => {
      const data = new FormData();
      data.append("file", file);
      data.append("target_column", "churn");
      data.append("display_name", file.name.replace(".csv", ""));
      const uploaded = await api.uploadDataset(data);
      setSelectedCurrent(uploaded.id);
      setMessage(`Uploaded ${uploaded.name}`);
    });
  }

  const metrics = activeModel?.metrics;
  const topFeatureBars =
    activeModel?.feature_importance.slice(0, 8).map((item) => ({
      label: item.feature,
      value: item.importance,
      tone: "teal" as const,
    })) ?? [];
  const driftBars =
    latestDrift?.feature_reports.slice(0, 8).map((item) => ({
      label: item.feature,
      value: item.score,
      tone: item.status === "drifted" ? ("rose" as const) : ("indigo" as const),
    })) ?? [];

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <Rocket size={22} />
          </div>
          <div>
            <span>Param Saxena</span>
            <strong>MLOps Monitor</strong>
          </div>
        </div>

        <nav className="nav-list" aria-label="Dashboard sections">
          <a href="#overview">
            <Activity size={17} /> Overview
          </a>
          <a href="#training">
            <Wand2 size={17} /> Training
          </a>
          <a href="#drift">
            <AlertTriangle size={17} /> Drift
          </a>
          <a href="#predictions">
            <BarChart3 size={17} /> Predictions
          </a>
        </nav>

        <section className="sidebar-panel">
          <span>Active model</span>
          <strong>{activeModel?.version ?? "Waiting"}</strong>
          <StatusBadge status={activeModel?.is_active ? "production" : "not trained"} tone={activeModel?.is_active ? "ok" : "warn"} />
        </section>
      </aside>

      <section className="workspace">
        <header className="topbar" id="overview">
          <div>
            <p className="eyebrow">Final-year B.Tech CSE Data Science project</p>
            <h1>MLOps Model Monitoring & Auto-Retraining Platform</h1>
            <span>{message}</span>
          </div>
          <button className="icon-button" onClick={loadAll} title="Refresh dashboard" aria-label="Refresh dashboard">
            <RefreshCcw size={18} />
          </button>
        </header>

        {error && (
          <div className="alert" role="alert">
            <AlertTriangle size={18} />
            {error}
          </div>
        )}

        <section className="metric-grid">
          <MetricCard label="Model F1" value={pct(metrics?.f1)} detail={`Accuracy ${pct(metrics?.accuracy)}`} icon={ShieldCheck} tone="teal" />
          <MetricCard label="Versions" value={`${summary?.counts.model_versions ?? models.length}`} detail="tracked in SQLite" icon={GitBranch} tone="indigo" />
          <MetricCard
            label="Drift score"
            value={(latestDrift?.drift_score ?? 0).toFixed(3)}
            detail={latestDrift?.should_retrain ? "retraining signal" : "within threshold"}
            icon={AlertTriangle}
            tone={latestDrift?.should_retrain ? "rose" : "amber"}
          />
          <MetricCard
            label="Predictions"
            value={`${summary?.counts.predictions ?? 0}`}
            detail={`${summary?.average_latency_ms ?? 0} ms avg latency`}
            icon={Activity}
            tone="amber"
          />
        </section>

        <section className="control-surface">
          <div className="field">
            <label>Baseline dataset</label>
            <select value={selectedBaseline ?? ""} onChange={(event) => setSelectedBaseline(Number(event.target.value))}>
              {datasets.map((dataset) => (
                <option key={dataset.id} value={dataset.id}>
                  {dataset.name}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>Current dataset</label>
            <select value={selectedCurrent ?? ""} onChange={(event) => setSelectedCurrent(Number(event.target.value))}>
              {datasets.map((dataset) => (
                <option key={dataset.id} value={dataset.id}>
                  {dataset.name}
                </option>
              ))}
            </select>
          </div>
          <button
            onClick={() =>
              runAction("train", async () => {
                if (!selectedBaseline) throw new Error("Select a training dataset first.");
                const dataset = datasets.find((item) => item.id === selectedBaseline);
                await api.trainModel(selectedBaseline, dataset?.target_column ?? "churn");
                setMessage("New model version trained");
              })
            }
            disabled={busy !== null}
          >
            <Play size={17} /> Train
          </button>
          <button
            onClick={() =>
              runAction("drift", async () => {
                requireSelection();
                const report = await api.detectDrift(selectedBaseline!, selectedCurrent!, activeModel?.version);
                setDriftReport(report);
                setMessage("Drift report created");
              })
            }
            disabled={busy !== null}
          >
            <AlertTriangle size={17} /> Drift
          </button>
          <button
            className="primary"
            onClick={() =>
              runAction("monitor", async () => {
                requireSelection();
                await api.runMonitor(selectedBaseline!, selectedCurrent!, activeModel?.version);
                setMessage("Monitoring cycle finished");
              })
            }
            disabled={busy !== null}
          >
            <RefreshCcw size={17} /> Monitor
          </button>
          <label className="upload-button">
            <Upload size={17} />
            Upload CSV
            <input type="file" accept=".csv" onChange={(event) => handleUpload(event.target.files?.[0])} />
          </label>
        </section>

        <div className="dashboard-grid">
          <section className="panel wide" id="training">
            <div className="panel-title">
              <div>
                <span>Training performance</span>
                <h2>{activeModel ? `${activeModel.model_name} ${activeModel.version}` : "No model yet"}</h2>
              </div>
              <StatusBadge status={activeModel?.trigger_reason ?? "not trained"} tone="neutral" />
            </div>
            <div className="score-strip">
              <div>
                <span>Accuracy</span>
                <strong>{pct(metrics?.accuracy)}</strong>
              </div>
              <div>
                <span>Precision</span>
                <strong>{pct(metrics?.precision)}</strong>
              </div>
              <div>
                <span>Recall</span>
                <strong>{pct(metrics?.recall)}</strong>
              </div>
              <div>
                <span>F1</span>
                <strong>{pct(metrics?.f1)}</strong>
              </div>
            </div>
            <ConfusionMatrix data={activeModel?.confusion_matrix} />
          </section>

          <section className="panel">
            <div className="panel-title">
              <div>
                <span>Feature importance</span>
                <h2>Top drivers</h2>
              </div>
            </div>
            {topFeatureBars.length ? <Bars items={topFeatureBars} /> : <div className="empty-state">Train a model to view feature importance.</div>}
          </section>

          <section className="panel" id="drift">
            <div className="panel-title">
              <div>
                <span>Data drift</span>
                <h2>{latestDrift?.should_retrain ? "Retraining recommended" : "Stable window"}</h2>
              </div>
              <StatusBadge status={`${latestDrift?.drifted_features.length ?? 0} drifted`} tone={latestDrift?.should_retrain ? "danger" : "ok"} />
            </div>
            {driftBars.length ? <Bars items={driftBars} maxValue={1} /> : <div className="empty-state">Run drift detection to compare datasets.</div>}
          </section>

          <section className="panel" id="predictions">
            <div className="panel-title">
              <div>
                <span>Prediction monitoring</span>
                <h2>Live request log</h2>
              </div>
              <button
                className="small-button"
                onClick={() =>
                  runAction("predictions", async () => {
                    if (!selectedCurrent) throw new Error("Select a dataset first.");
                    await api.demoPredictions(selectedCurrent, activeModel?.version);
                    setMessage("Demo predictions generated");
                  })
                }
                disabled={busy !== null}
              >
                <Activity size={15} /> Batch
              </button>
            </div>
            <div className="distribution">
              {Object.entries(summary?.prediction_distribution ?? {}).map(([label, count]) => (
                <div key={label}>
                  <span>Class {label}</span>
                  <strong>{count}</strong>
                </div>
              ))}
              {!Object.keys(summary?.prediction_distribution ?? {}).length && <div className="empty-state">No prediction traffic yet.</div>}
            </div>
            <div className="log-list">
              {summary?.recent_predictions.slice(0, 8).map((log) => (
                <div key={log.id} className="log-row">
                  <span>#{log.id}</span>
                  <strong>Class {log.prediction}</strong>
                  <small>{log.probability ? `${Math.round(log.probability * 100)}%` : "n/a"}</small>
                  <small>{formatDate(log.created_at)}</small>
                </div>
              ))}
            </div>
          </section>

          <section className="panel">
            <div className="panel-title">
              <div>
                <span>Manual prediction</span>
                <h2>Sample payload</h2>
              </div>
              <button
                className="small-button"
                onClick={() =>
                  runAction("predict", async () => {
                    const sample = await api.sampleInput(selectedCurrent ?? undefined);
                    const result = await api.predict(sample.features, activeModel?.version);
                    setPredictionResult(result);
                    setMessage("Prediction logged");
                  })
                }
                disabled={busy !== null}
              >
                <Play size={15} /> Predict
              </button>
            </div>
            <pre className="json-box">{predictionResult ? JSON.stringify(predictionResult, null, 2) : "No prediction submitted yet."}</pre>
          </section>

          <section className="panel wide">
            <div className="panel-title">
              <div>
                <span>Model registry</span>
                <h2>Version timeline</h2>
              </div>
            </div>
            <div className="version-list">
              {models.map((model) => (
                <div className="version-row" key={model.id}>
                  <span>{model.version}</span>
                  <strong>F1 {pct(model.metrics.f1)}</strong>
                  <small>{model.trigger_reason}</small>
                  <small>{formatDate(model.created_at)}</small>
                  <StatusBadge status={model.is_active ? "active" : "archived"} tone={model.is_active ? "ok" : "neutral"} />
                </div>
              ))}
            </div>
          </section>

          <section className="panel wide">
            <div className="panel-title">
              <div>
                <span>Dataset workspace</span>
                <h2>{selectedCurrentDataset?.name ?? "No dataset selected"}</h2>
              </div>
              <StatusBadge status={`${selectedCurrentDataset?.row_count ?? 0} rows`} tone="neutral" />
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    {preview?.columns.map((column) => (
                      <th key={column}>{column}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview?.records.map((row, index) => (
                    <tr key={index}>
                      {preview.columns.map((column) => (
                        <td key={column}>{String(row[column] ?? "")}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="panel wide">
            <div className="panel-title">
              <div>
                <span>Automation history</span>
                <h2>Monitor events</h2>
              </div>
            </div>
            <div className="event-list">
              {summary?.recent_events.map((event) => (
                <div className="event-row" key={event.id}>
                  <span>{event.action}</span>
                  <strong>Drift {event.drift_score.toFixed(3)}</strong>
                  <small>F1 {pct(event.current_f1)}</small>
                  <small>{event.notes}</small>
                  <small>{formatDate(event.created_at)}</small>
                </div>
              ))}
              {!summary?.recent_events.length && <div className="empty-state">Run the monitoring cycle to create automation events.</div>}
            </div>
          </section>
        </div>
      </section>
    </main>
  );
}
