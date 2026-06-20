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
evaluate.py      — accuracy, F1, confusion matrix, classification report
visualize.py     — confusion matrix heatmap + PCA decision boundary
predict.py       — CLI: load saved model + scaler, predict species
```

## How to Run

```bash
pip install -r requirements.txt
python train.py
python predict.py 5.1 3.5 1.4 0.2
```

## Dependencies

- Python 3
- scikit-learn
- numpy
- matplotlib
- seaborn
- joblib
