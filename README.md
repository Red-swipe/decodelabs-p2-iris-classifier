# PRO#2 — Iris Flower Classifier (KNN)

A supervised machine learning project that classifies Iris flower species using a K-Nearest Neighbors classifier. This is **Project 2** of the DecodeLabs AI Internship Track, progressing from rule-based logic (PRO#1) into statistical ML approaches.

## Overview

The Iris dataset (150 samples, 3 species) is a classic introductory benchmark for classification. This project trains a `KNeighborsClassifier` (k=5) on scaled features and evaluates it with a confusion matrix, F1 score, and accuracy.

## How to Run

```bash
pip install -r requirements.txt
python iris_classifier.py
```

## Versions

This section documents the progression of the project across branches. Since the more advanced work lives on unmerged branches, this table makes the full picture visible from `main`.

### v1 — `main` (baseline)

The original pipeline: loads the Iris dataset, applies `StandardScaler`, performs an 80/20 train/test split (`random_state=42`), trains a single `KNeighborsClassifier(k=5)`, and evaluates with a confusion matrix, weighted F1, and accuracy. Proves the core supervised-learning loop end-to-end.

### v2 — [`v2-algorithm-comparison`](https://github.com/Red-swipe/decodelabs-p2-iris-classifier/tree/v2-algorithm-comparison)

Expands to a real model comparison: 6 classifiers (KNN, Logistic Regression, SVM-RBF, Decision Tree, Random Forest, Gaussian Naive Bayes) evaluated with StratifiedKFold cross-validation instead of a single split, with GridSearchCV tuning KNN's `k`. Fixes a data-leakage bug present in the original pipeline (the scaler was fit on the full dataset before splitting; corrected to fit only on training data).

### v3 — [`v3-production-pipeline`](https://github.com/Red-swipe/decodelabs-p2-iris-classifier/tree/v3-production-pipeline) (recommended)

The most advanced version — supersedes v1 and v2. Modular architecture (`data_loader.py`, `models.py`, `train.py`, `evaluate.py`, `visualize.py`, `predict.py`). Hyperparameter tuning via GridSearchCV across all 6 models. Saved model + scaler artifacts via joblib. A CLI (`predict.py`) that classifies new, unseen flower measurements. Confusion matrix heatmap + PCA decision-boundary visualizations.

On the standard test split all 6 models tied at a perfect score (F1 = 1.0000), which meant the comparison had no real signal — so a **stress-test evaluation** was added (a data-scarcity scenario training on only 30% of the data plus a Gaussian-noise injection) to force actual separation between models. **Result: KNN won with a combined stress-test F1 of 0.9278**, staying strong under both data scarcity (0.9237) and noise (0.9319), while Decision Tree and Gaussian NB collapsed under noise (0.7612 each).

**v3 supersedes v2, which supersedes v1.** A reviewer should check the `v3-production-pipeline` branch for the complete picture.

## Dependencies

- Python 3
- scikit-learn
- numpy
- matplotlib
