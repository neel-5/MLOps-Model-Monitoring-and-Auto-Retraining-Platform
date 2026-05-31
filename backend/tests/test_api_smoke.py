from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_datasets_and_active_model_bootstrap():
    with TestClient(app) as client:
        datasets = client.get("/api/datasets")
        active_model = client.get("/api/models/active")

    assert datasets.status_code == 200
    assert len(datasets.json()) >= 1
    assert active_model.status_code == 200
    assert active_model.json()["metrics"]["f1"] >= 0
