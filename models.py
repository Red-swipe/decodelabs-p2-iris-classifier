from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


def get_model_configs():
    configs = [
        ("KNN", KNeighborsClassifier(), {"n_neighbors": list(range(1, 16))}),
        (
            "Logistic Regression",
            LogisticRegression(max_iter=1000, random_state=42),
            {"C": [0.01, 0.1, 1, 10, 100]},
        ),
        (
            "SVC (RBF)",
            SVC(random_state=42),
            {"C": [0.1, 1, 10], "gamma": ["scale", "auto"], "kernel": ["rbf"]},
        ),
        (
            "Decision Tree",
            DecisionTreeClassifier(random_state=42),
            {"max_depth": [3, 5, 7, 10, None]},
        ),
        (
            "Random Forest",
            RandomForestClassifier(random_state=42),
            {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]},
        ),
        ("Gaussian NB", GaussianNB(), {}),
    ]
    return configs
