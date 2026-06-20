import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from evaluate import evaluate_model


def plot_confusion_matrix(model, X_test, y_test, target_names, output_dir):
    _, acc, _, cm, _ = evaluate_model(model, X_test, y_test, target_names)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.title(f"Confusion Matrix (Accuracy={acc:.4f})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved confusion matrix plot: {path}")


def plot_decision_boundary(
    best_model, X_train, y_train, X_scaled, y, scaler, target_names, output_dir
):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    x_min, x_max = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
    y_min, y_max = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_full = pca.inverse_transform(grid)
    Z = best_model.predict(grid_full)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="Set1")
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1], c=y, cmap="Set1", edgecolor="k", s=60
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Decision Boundary (PCA-reduced)")
    handles, _ = scatter.legend_elements()
    plt.legend(handles, target_names, title="Species")
    plt.tight_layout()
    path = os.path.join(output_dir, "decision_boundary.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved decision boundary plot: {path}")
