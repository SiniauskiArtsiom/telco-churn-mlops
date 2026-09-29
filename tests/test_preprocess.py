"""Unit tests for preprocessing utilities."""
from __future__ import annotations

import pandas as pd

from app.preprocess import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
    clean_raw,
    split_features_target,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "customerID": "1",
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
                "TotalCharges": "29.85",
                "Churn": "No",
            },
            {
                "customerID": "2",
                "gender": "Male",
                "SeniorCitizen": 1,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 30,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "One year",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 90.0,
                "TotalCharges": " ",
                "Churn": "Yes",
            },
        ]
    )


def test_clean_raw_converts_total_charges():
    df = clean_raw(_sample_df())
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])
    assert df["TotalCharges"].isna().sum() == 1


def test_split_features_target():
    df = clean_raw(_sample_df())
    X, y = split_features_target(df)
    assert "Churn" not in X.columns
    assert "customerID" not in X.columns
    assert set(y.unique()) <= {0, 1}
    assert y.tolist() == [0, 1]


def test_preprocessor_fit_transform():
    df = clean_raw(_sample_df())
    X, _ = split_features_target(df)
    pre = build_preprocessor()
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == 2
    assert Xt.shape[1] > len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES) - 5
