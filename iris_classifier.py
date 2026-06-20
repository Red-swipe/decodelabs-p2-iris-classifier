# Load the Iris dataset from sklearn's built-in datasets
# This returns feature data (X) and target labels (y) for 150 flower samples
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Apply StandardScaler to normalize all 4 features (sepal/petal length and width)
# This ensures each feature has mean=0 and std=1, preventing scale dominance
# Tree-based models are invariant to feature scale (splits are threshold-based, not
# distance-based), but we apply scaling once for pipeline consistency across all models.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split scaled data into 80% training and 20% test sets
# shuffle=True randomizes the data, random_state=42 ensures reproducible splits
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=True, random_state=42
)

# Stratified 5-fold cross-validation to evaluate generalisation reliably
# StratifiedKFold preserves class proportions in each fold, important for small datasets
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- Hyperparameter tuning: find the best k for KNN ---
# GridSearchCV wraps cross_val_score internally, testing every k from 1 to 15
knn_param_grid = {"n_neighbors": list(range(1, 16))}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=cv, scoring="accuracy")
knn_grid.fit(X_train, y_train)
best_k = knn_grid.best_params_["n_neighbors"]

# --- Model definitions ---
# Each model represents a fundamentally different family of classifiers:
models = {
    "KNN (tuned)": KNeighborsClassifier(n_neighbors=best_k),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVC (RBF kernel)": SVC(kernel="rbf", random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gaussian Naive Bayes": GaussianNB(),
}

# Store results for the final ranked summary
results = []

print("=" * 62)
print("IRIS CLASSIFIER — ALGORITHM COMPARISON (v2)")
print("=" * 62)

for name, model in models.items():
    # 5-fold cross-validation on the TRAINING set only
    # cross_val_score returns an array of 5 accuracy scores, one per fold
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    # Fit on the full training set and predict on the held-out test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics on the test set
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)

    results.append((name, f1, acc, cv_mean, cv_std))

    print()
    print("-" * 62)
    print(f"  {name}")
    print("-" * 62)
    print(f"  CV accuracy (5-fold):     {cv_mean:.4f}  ±  {cv_std:.4f}")
    print(f"  Test accuracy:            {acc:.4f}")
    print(f"  Test F1 (weighted avg):   {f1:.4f}")
    print()
    if name.startswith("KNN"):
        print(f"  Best k found:            {best_k}")
    print("Confusion Matrix:")
    print(f"{'':>10}", end="")
    for tname in target_names:
        print(f"{tname:>12}", end="")
    print()
    for i, tname in enumerate(target_names):
        print(f"{tname:>10}", end="")
        for j in range(len(target_names)):
            print(f"{cm[i][j]:>12}", end="")
        print()

# --- Final ranked summary table by test weighted F1 ---
results.sort(key=lambda r: r[1], reverse=True)

print()
print("=" * 62)
print("  FINAL RANKING (by test weighted F1)")
print("=" * 62)
print(f"  {'Rank':<6}{'Model':<25}{'F1':<10}{'Accuracy':<10}{'CV (mean±std)':<15}")
print("  " + "-" * 62)
for rank, (name, f1, acc, cv_mean, cv_std) in enumerate(results, 1):
    print(f"  {rank:<6}{name:<25}{f1:<10.4f}{acc:<10.4f}{cv_mean:.4f} ± {cv_std:.4f}")
print("=" * 62)
print()
