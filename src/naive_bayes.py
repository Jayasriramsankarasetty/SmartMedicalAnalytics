from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .preprocessing import load_csv


def _select_text_and_label(df: pd.DataFrame) -> Tuple[str, str]:
    text_cols = [col for col in df.columns if df[col].dtype == "object"]
    if not text_cols:
        raise ValueError("No text columns found for classification.")

    name_priority = [
        col
        for col in text_cols
        if any(key in col.lower() for key in ("description", "symptom", "text"))
    ]
    candidate_text_cols = name_priority or text_cols

    avg_lengths = {
        col: df[col].astype(str).str.len().mean() for col in candidate_text_cols
    }
    text_col = max(avg_lengths, key=avg_lengths.get)

    label_candidates = [col for col in df.columns if col != text_col]
    if not label_candidates:
        raise ValueError("No label column found for text classification.")

    label_col = min(label_candidates, key=lambda c: df[c].nunique())
    return text_col, label_col


def run_naive_bayes(data_path: str) -> Tuple[Pipeline, Dict[str, float], Dict[str, str]]:
    df = load_csv(data_path)
    text_col, label_col = _select_text_and_label(df)

    X = df[text_col].astype(str)
    y = df[label_col]

    stratify_arg = None
    if y.value_counts().min() >= 2:
        stratify_arg = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_arg
    )

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("classifier", MultinomialNB()),
        ]
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {"accuracy": accuracy_score(y_test, preds)}
    extra = {"text_col": text_col, "label_col": label_col}
    return model, metrics, extra
