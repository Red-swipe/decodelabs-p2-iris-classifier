import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report


def evaluate_model(model, X_test, y_test, target_names):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    return y_pred, acc, f1, cm, report


def print_confusion_matrix(cm, target_names):
    n = len(target_names)
    print(f"{'':>10}", end="")
    for tname in target_names:
        print(f"{tname:>12}", end="")
    print()
    for i, tname in enumerate(target_names):
        print(f"{tname:>10}", end="")
        for j in range(n):
            print(f"{cm[i][j]:>12}", end="")
        print()


def evaluate_stress_small_train(model, train_frac=0.3, random_state=42):
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train_small, X_test_large, y_train_small, y_test_large = train_test_split(
        X, y, train_size=train_frac, shuffle=True, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_small_scaled = scaler.fit_transform(X_train_small)
    X_test_large_scaled = scaler.transform(X_test_large)
    model.fit(X_train_small_scaled, y_train_small)
    y_pred = model.predict(X_test_large_scaled)
    return f1_score(y_test_large, y_pred, average="weighted")


def evaluate_stress_noise(model, X_test, y_test, noise_std=0.3, random_state=42):
    rng = np.random.default_rng(random_state)
    X_test_noisy = X_test + rng.normal(0, noise_std, size=X_test.shape)
    y_pred = model.predict(X_test_noisy)
    return f1_score(y_test, y_pred, average="weighted")
