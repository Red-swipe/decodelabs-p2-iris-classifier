from data_loader import load_and_split_data
from models import get_model_configs
from evaluate import evaluate_model, print_confusion_matrix


def main():
    X_train, X_test, y_train, y_test, scaler, target_names, X_scaled = (
        load_and_split_data()
    )
    model_configs = get_model_configs()

    print("Iris Classifier — Modular Pipeline (v3)")
    print(f"Loaded {len(model_configs)} model configurations")
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    name, model, param_grid = model_configs[0]
    model.fit(X_train, y_train)
    y_pred, acc, f1, cm, report = evaluate_model(model, X_test, y_test, target_names)
    print(f"\nSanity check — {name}: accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
