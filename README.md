# Smart Medical Analytics using Supervised and Unsupervised Learning

## Project Overview
This project demonstrates core supervised and unsupervised machine learning techniques using public healthcare datasets. It covers preprocessing, regression, classification, text classification, clustering, and dimensionality reduction, with results presented in a Streamlit dashboard.

## Features
- Data preprocessing with missing value handling, encoding, scaling, and train-test split
- Multiple Linear Regression to predict medical treatment cost
- Logistic Regression for disease risk prediction
- Naive Bayes text classification for symptom descriptions
- K-Means, Hierarchical Clustering, and DBSCAN for patient segmentation
- PCA for 2D visualization of clusters
- Streamlit dashboard with eight interactive pages

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset Details
Place the datasets in the `data/` directory with these exact filenames:
- `insurance.csv` (target: `charges`)
- `heart.csv`
- `dataset.csv`
- `symptom_Description.csv`
- `symptom_precaution.csv`
- `Symptom-severity.csv`

## Execution Steps
```bash
streamlit run app.py
```

## Results
The dashboard shows model performance metrics, confusion matrices, clustering plots, dendrograms, DBSCAN anomaly detection, and PCA visualizations. If expected target columns are missing, the app surfaces a clear error explaining what is required.
