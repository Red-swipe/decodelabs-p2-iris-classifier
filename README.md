# Project 2 — Iris Flower Classifier (KNN)

A supervised machine learning project that classifies Iris flower species using a K-Nearest Neighbors classifier. This is **Project 2** of the DecodeLabs AI Internship Track, progressing from rule-based logic (PRO#1) into statistical ML approaches.

## Overview

The Iris dataset (150 samples, 3 species) is a classic introductory benchmark for classification. This project trains a `KNeighborsClassifier` (k=5) on scaled features and evaluates it with a confusion matrix, F1 score, and accuracy.

## How to Run

```bash
pip install -r requirements.txt
python iris_classifier.py
```

## Versions

This section documents the progression of the project across branches. Since the more advanced work lives on unmerged branches, this section makes the full picture visible from `main`.

### 1. `main` (v1) — Baseline pipeline

Load Iris dataset, `StandardScaler`, 80/20 train/test split (`random_state=42`), single `KNeighborsClassifier(k=5)`, confusion matrix + weighted F1 + accuracy. Proves the core supervised learning loop end to end.

### 2. [`v2-algorithm-comparison`](https://github.com/Red-swipe/decodelabs-p2-iris-classifier/tree/v2-algorithm-comparison)

Expands to a real model comparison: 6 classifiers (KNN, Logistic Regression, SVM-RBF, Decision Tree, Random Forest, Gaussian Naive Bayes) evaluated with StratifiedKFold cross-validation instead of a single split, with GridSearchCV tuning KNN's k. Fixes a data-leakage bug present in the original pipeline (scaler was fit on the full dataset before splitting; corrected to fit only on training data).

### 3. [`v3-production-pipeline`](https://github.com/Red-swipe/decodelabs-p2-iris-classifier/tree/v3-production-pipeline)

The most advanced version, supersedes v1 and v2. Modular architecture (`data_loader.py`, `models.py`, `train.py`, `evaluate.py`, `visualize.py`, `predict.py`). Hyperparameter tuning via GridSearchCV across all 6 models. Saved model + scaler artifacts via joblib. A CLI (`predict.py`) that classifies new, unseen flower measurements. Confusion matrix heatmap + PCA decision boundary visualizations. Most importantly: on the standard test split all 6 models tied at a perfect score, which meant the comparison had no real signal — so a stress-test evaluation was added (small-data scenario + Gaussian noise injection) to force actual separation between models. Result: KNN won with a combined stress-test F1 of 0.9278, staying strong under both data scarcity (0.9237) and noise (0.9319), while Decision Tree and Gaussian NB collapsed under noise (0.7612 each). This is the version that should be referenced as the project's main deliverable.

**v3 supersedes v2, which supersedes v1.** A reviewer should check the `v3-production-pipeline` branch for the complete picture.

## Dependencies

- Python 3
- scikit-learn
- numpy
- matplotlib
