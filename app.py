from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src import classification, clustering, naive_bayes, preprocessing, regression, visualization


DATA_DIR = Path("data")


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    return preprocessing.load_csv(str(path))


def dataset_overview() -> None:
    st.header("Dataset Overview")

    datasets = {
        "insurance.csv": DATA_DIR / "insurance.csv",
        "heart.csv": DATA_DIR / "heart.csv",
        "dataset.csv": DATA_DIR / "dataset.csv",
    }

    for name, path in datasets.items():
        if not path.exists():
            st.warning(f"Missing dataset: {name}")
            continue
        df = load_data(path)
        with st.expander(f"📄 {name}", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")
    
    with st.expander("📋 Symptom Precautions", expanded=False):
        precaution_path = DATA_DIR / "symptom_precaution.csv"
        if precaution_path.exists():
            precautions = load_data(precaution_path)
            st.dataframe(precautions.head(20), use_container_width=True)

    with st.expander("📊 Symptom Severity", expanded=False):
        severity_path = DATA_DIR / "Symptom-severity.csv"
        if severity_path.exists():
            severity = load_data(severity_path)
            st.dataframe(severity.head(20), use_container_width=True)


def regression_results() -> None:
    st.header("Multiple Linear Regression")
    try:
        _, metrics = regression.run_regression(str(DATA_DIR / "insurance.csv"))
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MAE", f"${metrics['mae']:,.0f}")
        with col2:
            st.metric("RMSE", f"${metrics['rmse']:,.0f}")
        with col3:
            st.metric("R² Score", f"{metrics['r2']:.3f}")
    except ValueError as exc:
        st.error(str(exc))


def disease_prediction() -> None:
    st.header("Disease Prediction (Logistic Regression)")
    try:
        _, metrics, extra = classification.run_logistic(
            str(DATA_DIR / "dataset.csv"), str(DATA_DIR / "heart.csv")
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.1%}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.1%}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("F1 Score", f"{metrics['f1']:.3f}")
        with col2:
            st.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
        with col3:
            st.metric("Dataset", extra["target_col"])
        
        st.markdown("---")
        st.markdown("**Confusion Matrix**")
        fig = visualization.plot_confusion_matrix(extra["confusion_matrix"])
        st.pyplot(fig, use_container_width=True)
    except ValueError as exc:
        st.error(str(exc))


def text_classification() -> None:
    st.header("Text Classification (Naive Bayes)")
    try:
        _, metrics, extra = naive_bayes.run_naive_bayes(
            str(DATA_DIR / "symptom_Description.csv")
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        with col2:
            st.metric("Text Column", extra["text_col"])
        with col3:
            st.metric("Label Column", extra["label_col"])
    except ValueError as exc:
        st.error(str(exc))


def kmeans_clustering() -> None:
    st.header("K-Means Clustering")
    df = load_data(DATA_DIR / "heart.csv")
    clustering_data = clustering.prepare_clustering_data(df)

    k_values = list(range(2, 11))
    inertias = clustering.kmeans_elbow(clustering_data.features, k_values)
    optimal_k = clustering.find_elbow(k_values, inertias)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Optimal K", optimal_k)
    
    st.markdown("**Elbow Method**")
    st.pyplot(visualization.plot_elbow(k_values, inertias), use_container_width=True)

    labels = clustering.run_kmeans(clustering_data.features, optimal_k)
    points = clustering.run_pca(clustering_data.features)
    st.markdown("**Cluster Visualization (2D PCA)**")
    st.pyplot(
        visualization.plot_clusters(points, labels, "K-Means Clusters"),
        use_container_width=True,
    )


def hierarchical_clustering() -> None:
    st.header("Hierarchical Clustering")
    df = load_data(DATA_DIR / "heart.csv")
    clustering_data = clustering.prepare_clustering_data(df)

    labels = clustering.run_hierarchical(clustering_data.features, n_clusters=4)
    points = clustering.run_pca(clustering_data.features)
    
    st.markdown("**Cluster Visualization (2D PCA)**")
    st.pyplot(
        visualization.plot_clusters(points, labels, "Hierarchical Clusters"),
        use_container_width=True,
    )

    st.markdown("**Dendrogram**")
    linkage_matrix = clustering.compute_linkage(clustering_data.features)
    st.pyplot(visualization.plot_dendrogram(linkage_matrix), use_container_width=True)


def dbscan_clustering() -> None:
    st.header("DBSCAN")
    df = load_data(DATA_DIR / "heart.csv")
    clustering_data = clustering.prepare_clustering_data(df)

    col1, col2 = st.columns(2)
    with col1:
        eps = st.number_input("Epsilon", min_value=0.1, max_value=3.0, value=0.5, step=0.1)
    with col2:
        min_samples = st.number_input("Min samples", min_value=3, max_value=15, value=5, step=1)

    labels = clustering.run_dbscan(clustering_data.features, eps=eps, min_samples=min_samples)
    points = clustering.run_pca(clustering_data.features)

    noise_count = int(np.sum(labels == -1))
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Noise Points", noise_count)
    with col2:
        st.metric("Core Clusters", int(np.unique(labels).size - (1 if -1 in labels else 0)))
    
    st.pyplot(visualization.plot_dbscan(points, labels), use_container_width=True)


def pca_visualization() -> None:
    st.header("PCA Visualization")
    df = load_data(DATA_DIR / "heart.csv")
    clustering_data = clustering.prepare_clustering_data(df)

    k_values = list(range(2, 8))
    inertias = clustering.kmeans_elbow(clustering_data.features, k_values)
    optimal_k = clustering.find_elbow(k_values, inertias)

    labels = clustering.run_kmeans(clustering_data.features, optimal_k)
    points = clustering.run_pca(clustering_data.features)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Clusters", optimal_k)
    
    st.markdown("**Data Projection in 2D Space**")
    st.pyplot(
        visualization.plot_clusters(points, labels, "PCA Projection with K-Means"),
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Smart Medical Analytics",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
        <style>
        body { background-color: #ffffff; }
        .main { background-color: #ffffff; }
        .sidebar .sidebar-content { background-color: #f8f9fa; }
        h1, h2, h3 { color: #1a1a1a; }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("📊 Smart Medical Analytics")
    st.markdown("---")

    pages = {
        "Dataset Overview": dataset_overview,
        "Regression Results": regression_results,
        "Disease Prediction": disease_prediction,
        "Text Classification": text_classification,
        "K-Means Clustering": kmeans_clustering,
        "Hierarchical Clustering": hierarchical_clustering,
        "DBSCAN": dbscan_clustering,
        "PCA Visualization": pca_visualization,
    }

    st.sidebar.markdown("### Navigation")
    selection = st.sidebar.radio("Select Module", list(pages.keys()), label_visibility="collapsed")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Smart Medical Analytics**\n\nSupervised & Unsupervised Learning")
    
    pages[selection]()


if __name__ == "__main__":
    main()
