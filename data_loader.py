from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_and_split_data(test_size=0.2, random_state=42):
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=True, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_full_scaled = scaler.transform(X)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, target_names, X_full_scaled, y
