from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NAME_CANDIDATES = (
    "target",
    "label",
    "disease",
    "diagnosis",
    "risk",
    "outcome",
)


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame with basic validation."""
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Dataset is empty: {path}")
    return df


def _is_binary(series: pd.Series) -> bool:
    values = series.dropna().unique()
    return len(values) == 2


def get_target_column(
    df: pd.DataFrame,
    preferred: Optional[str] = None,
    binary_only: bool = False,
) -> Optional[str]:
    """Infer a target column by name or by binary values."""
    if preferred and preferred in df.columns:
        if not binary_only or _is_binary(df[preferred]):
            return preferred

    lower_cols = {col.lower(): col for col in df.columns}
    for candidate in NAME_CANDIDATES:
        if candidate in lower_cols:
            col = lower_cols[candidate]
            if not binary_only or _is_binary(df[col]):
                return col

    if binary_only:
        for col in df.columns:
            if _is_binary(df[col]):
                return col
    return None


def split_features_target(
    df: pd.DataFrame, target_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False,
):
    stratify_arg = y if stratify else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )


def list_missing_columns(df: pd.DataFrame, required: Iterable[str]) -> list[str]:
    return [col for col in required if col not in df.columns]
