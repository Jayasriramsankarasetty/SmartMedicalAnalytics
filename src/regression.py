from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor, load_csv, split_data, split_features_target


def run_regression(data_path: str, target_col: str = "charges") -> Tuple[Pipeline, Dict[str, float]]:
    df = load_csv(data_path)
    if target_col not in df.columns:
        raise ValueError(f"Missing target column for regression: {target_col}")

    X, y = split_features_target(df, target_col)
    X_train, X_test, y_train, y_test = split_data(X, y)

    preprocessor = build_preprocessor(X_train)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": np.sqrt(mean_squared_error(y_test, preds)),
        "r2": r2_score(y_test, preds),
    }
    return model, metrics
