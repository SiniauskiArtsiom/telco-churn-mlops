"""API tests for the Telco Churn service."""
from __future__ import annotations


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert "model_name" in body
    assert isinstance(body["features"], list)
    assert "tenure" in body["features"]
    assert "Contract" in body["features"]


def test_predict_risky_customer(client, risky_customer):
    r = client.post("/predict", json=risky_customer)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_probability"] > 0.5
    assert body["churn_prediction"] is True
    assert body["threshold"] == 0.5


def test_predict_loyal_customer(client, loyal_customer):
    r = client.post("/predict", json=loyal_customer)
    assert r.status_code == 200
    body = r.json()
    assert body["churn_probability"] < 0.5
    assert body["churn_prediction"] is False


def test_predict_missing_field(client, risky_customer):
    payload = dict(risky_customer)
    payload.pop("tenure")
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_invalid_tenure(client, risky_customer):
    payload = dict(risky_customer)
    payload["tenure"] = -5
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_invalid_category(client, risky_customer):
    payload = dict(risky_customer)
    payload["Contract"] = "Weekly"
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "http_requests_total" in r.text or "python_info" in r.text
