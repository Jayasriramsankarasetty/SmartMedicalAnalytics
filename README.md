# Smart Medical Analytics

## Project Overview
Comprehensive machine learning analysis on healthcare datasets demonstrating supervised and unsupervised learning techniques. Covers regression, classification, text classification, clustering, and dimensionality reduction using a single consolidated Jupyter notebook.

## Features
- **Multiple Linear Regression**: Predict insurance charges from demographic/health features
- **Logistic Regression**: Predict heart disease risk (binary classification)
- **Naive Bayes Text Classification**: Classify medical symptom descriptions
- **K-Means Clustering**: Segment patients into 3 distinct groups
- **Hierarchical Clustering**: Alternative patient segmentation using agglomerative approach
- **PCA**: Reduce 14-dimensional heart data to 2D for visualization
- **Data Preprocessing**: Missing value handling, feature scaling, encoding, train-test split

## Datasets
Three healthcare datasets (located in `data/` folder):
1. **insurance.csv** (1,338 records × 7 features)
   - Target: Medical charges
   - Features: age, BMI, children, sex, smoker, region
   
2. **heart.csv** (1,025 records × 14 features)
   - Target: Heart disease presence (binary)
   - Features: 13 medical metrics (age, blood pressure, cholesterol, etc.)
   
3. **symptom_Description.csv** (41 records × 2 columns)
   - Target: Symptom category
   - Features: Text descriptions

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
Open and run the Jupyter notebook:
```bash
jupyter notebook SmartMedicalAnalytics.ipynb
```

Or with JupyterLab:
```bash
jupyter lab SmartMedicalAnalytics.ipynb
```

## Results Summary
| Model | Metric | Score |
|-------|--------|-------|
| Linear Regression | R² Score | 0.784 |
| Linear Regression | MAE | $4,181 |
| Logistic Regression | Accuracy | 70.2% |
| Logistic Regression | ROC AUC | 0.811 |
| K-Means Clustering | Clusters | 3 groups |
| Hierarchical Clustering | Clusters | 3 groups |
| PCA | Variance Explained | 36.8% (2 components) |
| Text Classification | Accuracy | Limited due to small dataset |
