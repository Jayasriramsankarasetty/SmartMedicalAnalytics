from __future__ import annotations

from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram


sns.set_theme(style="whitegrid")


def plot_elbow(k_values: Iterable[int], inertias: Iterable[float]) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.plot(list(k_values), list(inertias), marker="o")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method")
    return fig


def plot_clusters(points: np.ndarray, labels: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots()
    scatter = ax.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10", s=30)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    return fig


def plot_dbscan(points: np.ndarray, labels: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots()
    noise_mask = labels == -1
    ax.scatter(points[~noise_mask, 0], points[~noise_mask, 1], c=labels[~noise_mask], cmap="tab10", s=30)
    ax.scatter(points[noise_mask, 0], points[noise_mask, 1], c="red", s=40, label="Noise")
    ax.set_title("DBSCAN Clusters and Noise")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend()
    return fig


def plot_confusion_matrix(cm: np.ndarray, labels: Optional[list[str]] = None) -> plt.Figure:
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    return fig


def plot_dendrogram(linkage_matrix: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    dendrogram(linkage_matrix, ax=ax, truncate_mode="lastp", p=30)
    ax.set_title("Hierarchical Clustering Dendrogram")
    ax.set_xlabel("Cluster size")
    ax.set_ylabel("Distance")
    return fig
