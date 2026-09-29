"""Smoke test the running API."""
from __future__ import annotations

import sys
import httpx

BASE = "http://localhost:8000"

SAMPLE = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


def main() -> int:
    with httpx.Client(base_url=BASE, timeout=10.0) as client:
        r = client.get("/health")
        print("GET /health ->", r.status_code, r.json())
        if r.status_code != 200:
            return 1

        r = client.get("/model-info")
        print("GET /model-info ->", r.status_code, r.json().get("model_name"))

        r = client.post("/predict", json=SAMPLE)
        print("POST /predict ->", r.status_code, r.json())
        if r.status_code != 200:
            return 1

        r = client.get("/metrics")
        print("GET /metrics ->", r.status_code, "body length:", len(r.text))
    print("\nSmoke test passed ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
