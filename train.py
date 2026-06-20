import os
import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from data_loader import load_and_split_data
from models import get_model_configs
from evaluate import evaluate_model, print_confusion_matrix


MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test, scaler, target_names, X_scaled, y = (
        load_and_split_data()
    )
    model_configs = get_model_configs()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    best_entry = None

    print("=" * 80)
    print("IRIS CLASSIFIER — PRODUCTION PIPELINE (v3)")
    print("=" * 80)

    for name, estimator, param_grid in model_configs:
        grid = GridSearchCV(estimator, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
        grid.fit(X_train, y_train)

        best = grid.best_estimator_
        y_pred, acc, f1, cm, report = evaluate_model(
            best, X_test, y_test, target_names
        )
        results.append((name, f1, acc, grid.best_params_))

        if best_entry is None or f1 > best_entry[1]:
            best_entry = (name, f1, best)

        print()
        print("-" * 80)
        print(f"  {name}")
        print("-" * 80)
        if grid.best_params_:
            print(f"  Best params:             {grid.best_params_}")
        print(f"  CV accuracy (5-fold):     {grid.best_score_:.4f}")
        print(f"  Test accuracy:            {acc:.4f}")
        print(f"  Test F1 (weighted avg):   {f1:.4f}")
        print()
        print("  Confusion Matrix:")
        print_confusion_matrix(cm, target_names)
        print()

    results.sort(key=lambda r: r[1], reverse=True)

    print()
    print("=" * 110)
    print("  FINAL RANKING (by test weighted F1)")
    print("=" * 110)
    print(f"  {'Rank':<6}{'Model':<25}{'F1':<10}{'Accuracy':<10}{'Best Params':<50}")
    print("  " + "-" * 106)
    for rank, (name, f1, acc, best_params) in enumerate(results, 1):
        params_str = str(best_params) if best_params else "(none)"
        print(f"  {rank:<6}{name:<25}{f1:<10.4f}{acc:<10.4f}{params_str:<50}")

    best_name, best_f1, best_model = best_entry
    print()
    print("=" * 80)
    print(f"  BEST MODEL: {best_name}  (F1 = {best_f1:.4f})")
    print("=" * 80)

    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    print(f"  Saved best_model.joblib and scaler.joblib to {MODELS_DIR}/")

    try:
        from visualize import plot_confusion_matrix, plot_decision_boundary

        plot_confusion_matrix(best_model, X_test, y_test, target_names, PLOTS_DIR)
        plot_decision_boundary(
            best_model, X_train, y_train, X_scaled, y, scaler, target_names, PLOTS_DIR
        )
        print(f"  Plots saved to {PLOTS_DIR}/")
    except ImportError:
        print("  Skipping visualization (visualize.py not yet created)")


if __name__ == "__main__":
    main()
