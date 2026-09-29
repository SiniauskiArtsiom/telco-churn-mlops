"""Model loading and inference helpers."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("model/churn_model.pkl")
METRICS_PATH = Path("model/metrics.json")


@lru_cache(maxsize=1)
def load_model():
    """Load the trained model once and cache it."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_metrics() -> dict | None:
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text())


def get_model_name() -> str:
    model = load_model()
    return model.__class__.__name__


def predict_proba(features: dict) -> float:
    """Return churn probability for a single customer."""
    model = load_model()
    df = pd.DataFrame([features])

    # CatBoost требует NaN-заполнение в категориальных
    if model.__class__.__name__ == "CatBoostClassifier":
        cat_features = [c for c in df.columns if df[c].dtype == "object"]
        df[cat_features] = df[cat_features].fillna("NA")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    proba = model.predict_proba(df)[:, 1][0]
    return float(proba)
