"""Train baseline and CatBoost models for Telco Churn, save the best one."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.preprocess import build_preprocessor, clean_raw, split_features_target

RANDOM_STATE = 42
DATA_PATH = Path("data/telco_churn.csv")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

METRICS_PATH = MODEL_DIR / "metrics.json"
BEST_MODEL_PATH = MODEL_DIR / "churn_model.pkl"


def evaluate(name: str, model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "model": name,
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "f1": round(float(f1_score(y_test, preds)), 4),
        "report": classification_report(y_test, preds, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }
    print(f"\n=== {name} ===")
    print(f"ROC-AUC: {metrics['roc_auc']}")
    print(f"F1:      {metrics['f1']}")
    print(classification_report(y_test, preds, digits=3))
    return metrics


def main() -> None:
    print("Loading data...")
    df = clean_raw(pd.read_csv(DATA_PATH))
    X, y = split_features_target(df)
    print(f"X shape: {X.shape}, positive rate: {y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    preprocessor = build_preprocessor()

    # --- Baseline: Logistic Regression ---
    logreg = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                       random_state=RANDOM_STATE)),
        ]
    )
    logreg.fit(X_train, y_train)
    logreg_metrics = evaluate("LogisticRegression", logreg, X_test, y_test)

    # --- CatBoost (с категориальными фичами) ---
    cat_features = [c for c in X_train.columns if X_train[c].dtype == "object"]
    X_train_cb = X_train.copy()
    X_test_cb = X_test.copy()
    X_train_cb[cat_features] = X_train_cb[cat_features].fillna("NA")
    X_test_cb[cat_features] = X_test_cb[cat_features].fillna("NA")
    # TotalCharges: median impute для CatBoost
    median_tc = X_train_cb["TotalCharges"].median()
    X_train_cb["TotalCharges"] = X_train_cb["TotalCharges"].fillna(median_tc)
    X_test_cb["TotalCharges"] = X_test_cb["TotalCharges"].fillna(median_tc)

    catboost = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        verbose=100,
        cat_features=cat_features,
    )
    catboost.fit(X_train_cb, y_train, eval_set=(X_test_cb, y_test), use_best_model=True)
    catboost_metrics = evaluate("CatBoost", catboost, X_test_cb, y_test)

    # --- Выбор лучшей модели ---
    results = [logreg_metrics, catboost_metrics]
    best = max(results, key=lambda m: m["roc_auc"])
    print(f"\nBest model: {best['model']} (ROC-AUC = {best['roc_auc']})")

    if best["model"] == "CatBoost":
        joblib.dump(catboost, BEST_MODEL_PATH)
    else:
        joblib.dump(logreg, BEST_MODEL_PATH)

    METRICS_PATH.write_text(json.dumps(results, indent=2))
    print(f"Saved model -> {BEST_MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
