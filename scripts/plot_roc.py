"""Plot ROC curve for the best saved model."""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from app.preprocess import clean_raw, split_features_target

RANDOM_STATE = 42

df = clean_raw(pd.read_csv("data/telco_churn.csv"))
X, y = split_features_target(df)
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

model = joblib.load("model/churn_model.pkl")

# CatBoost не принимает NaN в категориальных — подготовим как в train.py
if model.__class__.__name__ == "CatBoostClassifier":
    cat_features = [c for c in X_test.columns if X_test[c].dtype == "object"]
    X_test = X_test.copy()
    X_test[cat_features] = X_test[cat_features].fillna("NA")
    X_test["TotalCharges"] = X_test["TotalCharges"].fillna(X_test["TotalCharges"].median())

proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, proba)
auc = roc_auc_score(y_test, proba)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Telco Churn")
plt.legend(loc="lower right")
plt.tight_layout()

out = Path("notebooks/figures/06_roc_curve.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=120)
print(f"Saved {out} | AUC={auc:.4f}")
