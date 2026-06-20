# PRO#2 — Iris Flower Classifier

A supervised machine learning project that classifies Iris flower species using 6 ML models with hyperparameter tuning, cross-validation, and a modular pipeline. This is **Project 2** of the DecodeLabs AI Internship Track.

## Versions

| Branch | Description |
|---|---|
| [`main`](../../tree/main) | Baseline KNN (k=5) classifier |
| [`v2-algorithm-comparison`](../../tree/v2-algorithm-comparison) | 6 models with StratifiedKFold CV and ranking |
| [`v3-production-pipeline`](../../tree/v3-production-pipeline) | Modular pipeline with GridSearchCV, model persistence, visualization, and prediction CLI |

## Pipeline Overview

```
iris_classifier.py (v1/v2 monolithic)
        ↓
data_loader.py   — load, scale, train/test split
models.py        — 6 model configs + hyperparameter grids
train.py         — GridSearchCV, evaluation, ranking, save artifacts
evaluate.py      — accuracy, F1, confusion matrix, classification report, stress tests
visualize.py     — confusion matrix heatmap + PCA decision boundary
predict.py       — CLI: load saved model + scaler, predict species
```

## How to Run

```bash
pip install -r requirements.txt
python train.py
python predict.py 5.1 3.5 1.4 0.2
```

## The Iris Ceiling Problem

On the Iris dataset, all 6 models achieve **near-perfect accuracy (≈F1=1.0000)** on the test set. This makes standard metrics meaningless for model selection — every classifier looks like a winner.

The root cause is that Iris is an **easy, near-linearly-separable dataset**. Even simple models like Logistic Regression can classify it perfectly.

## Data Leakage Fix (v3)

The original `data_loader.py` fitted `StandardScaler` on the full dataset before splitting:

```python
# Before (leakage)
X_scaled = scaler.fit_transform(X)
X_train, X_test, ... = train_test_split(X_scaled, ...)   # ❌ test stats leaked
```

This was fixed to split raw data first, then fit the scaler only on X_train:

```python
# After (correct)
X_train, X_test, ... = train_test_split(X, ...)
X_train_scaled = scaler.fit_transform(X_train)            # ✅ only training data
X_test_scaled = scaler.transform(X_test)                  # ✅ transform, not fit
```

The decision-boundary plot still receives correctly ordered data via `scaler.transform(X)`.

## Stress-Test Methodology

To create a meaningful comparison, the pipeline now runs two stress tests on every model:

1. **Small Training Set** — trains each model on only 30% of the data and evaluates on 70%. Tests how well a model generalizes from few examples.
2. **Noise Injection** — adds Gaussian noise (std=0.3) to the scaled test features. Tests how robust a model is to corrupted inputs.

The **stress-test winner** is the model with the highest average F1 across both stress scenarios. This model is saved as `best_model.joblib`.

## Dependencies

- Python 3
- scikit-learn
- numpy
- matplotlib
- seaborn
- joblib
