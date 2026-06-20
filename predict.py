import os
import sys
import argparse
import numpy as np
import joblib


def main():
    parser = argparse.ArgumentParser(
        description="Predict Iris species from sepal/petal measurements"
    )
    parser.add_argument("sepal_length", type=float, help="Sepal length (cm)")
    parser.add_argument("sepal_width", type=float, help="Sepal width (cm)")
    parser.add_argument("petal_length", type=float, help="Petal length (cm)")
    parser.add_argument("petal_width", type=float, help="Petal width (cm)")
    args = parser.parse_args()

    models_dir = os.path.join(os.path.dirname(__file__), "models")
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    scaler_path = os.path.join(models_dir, "scaler.joblib")

    if not os.path.exists(best_model_path) or not os.path.exists(scaler_path):
        print(
            "Error: trained model not found. Run `python train.py` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    model = joblib.load(best_model_path)
    scaler = joblib.load(scaler_path)

    features = np.array(
        [[args.sepal_length, args.sepal_width, args.petal_length, args.petal_width]]
    )
    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)[0]

    target_names = ["setosa", "versicolor", "virginica"]
    print(f"Predicted species: {target_names[pred]}")


if __name__ == "__main__":
    main()
