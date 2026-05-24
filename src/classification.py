from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from .preprocessing import (
    build_preprocessor,
    get_target_column,
    load_csv,
    split_data,
    split_features_target,
)


def _select_dataset(primary_path: str, fallback_path: str) -> Tuple[pd.DataFrame, str, str]:
    df_primary = load_csv(primary_path)
    target = get_target_column(df_primary, binary_only=True)
    if target:
        return df_primary, primary_path, target

    df_fallback = load_csv(fallback_path)
    target = get_target_column(df_fallback, binary_only=True)
    if target:
        return df_fallback, fallback_path, target

    raise ValueError("No binary target column found in provided datasets.")


def run_logistic(primary_path: str, fallback_path: str) -> Tuple[Pipeline, Dict[str, float], Dict[str, object]]:
    df, used_path, target_col = _select_dataset(primary_path, fallback_path)

    X, y = split_features_target(df, target_col)
    X_train, X_test, y_train, y_test = split_data(X, y, stratify=True)

    preprocessor = build_preprocessor(X_train)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
    }

    if len(np.unique(y_test)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_test, probas)
    else:
        metrics["roc_auc"] = float("nan")

    extra = {
        "confusion_matrix": confusion_matrix(y_test, preds),
        "dataset_path": used_path,
        "target_col": target_col,
    }
    return model, metrics, extra
