from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor, get_target_column


@dataclass
class ClusteringData:
    features: np.ndarray
    feature_names: list[str]


def prepare_clustering_data(df: pd.DataFrame) -> ClusteringData:
    target_col = get_target_column(df, binary_only=True)
    if target_col and target_col in df.columns:
        X = df.drop(columns=[target_col])
    else:
        X = df

    preprocessor = build_preprocessor(X)
    pipeline = Pipeline([("preprocessor", preprocessor)])
    features = pipeline.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out().tolist()
    return ClusteringData(features=features, feature_names=feature_names)


def kmeans_elbow(features: np.ndarray, k_values: Iterable[int]) -> list[float]:
    inertias = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(features)
        inertias.append(model.inertia_)
    return inertias


def find_elbow(k_values: list[int], inertias: list[float]) -> int:
    if len(k_values) < 3:
        return k_values[0]

    points = np.array(list(zip(k_values, inertias)))
    start, end = points[0], points[-1]
    line_vec = end - start
    line_vec = line_vec / np.linalg.norm(line_vec)
    distances = []
    for point in points:
        vec = point - start
        proj = np.dot(vec, line_vec)
        closest = start + proj * line_vec
        distances.append(np.linalg.norm(point - closest))
    return k_values[int(np.argmax(distances))]


def run_kmeans(features: np.ndarray, n_clusters: int) -> np.ndarray:
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return model.fit_predict(features)


def run_hierarchical(features: np.ndarray, n_clusters: int) -> np.ndarray:
    model = AgglomerativeClustering(n_clusters=n_clusters)
    return model.fit_predict(features)


def run_dbscan(features: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(features)


def run_pca(features: np.ndarray, n_components: int = 2) -> np.ndarray:
    model = PCA(n_components=n_components, random_state=42)
    return model.fit_transform(features)


def compute_linkage(features: np.ndarray) -> np.ndarray:
    return linkage(features, method="ward")
