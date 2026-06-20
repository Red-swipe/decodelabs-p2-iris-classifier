# Load the Iris dataset from sklearn's built-in datasets
# This returns feature data (X) and target labels (y) for 150 flower samples
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Apply StandardScaler to normalize all 4 features (sepal/petal length and width)
# This ensures each feature has mean=0 and std=1, preventing scale dominance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split scaled data into 80% training and 20% test sets
# shuffle=True randomizes the data, random_state=42 ensures reproducible splits
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=True, random_state=42
)

# Train a KNeighborsClassifier with n_neighbors=5
# The model classifies points based on the majority class of their 5 nearest neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict on the test set using the trained model
# The model assigns a class label (0, 1, or 2) to each test sample
y_pred = knn.predict(X_test)

# Print a clearly labeled Confusion Matrix
# Rows = actual classes, Columns = predicted classes; diagonals are correct predictions
cm = confusion_matrix(y_test, y_pred)
print("=" * 50)
print("IRIS CLASSIFIER - RESULTS")
print("=" * 50)
print()
print("Confusion Matrix:")
print(f"{'':>10}", end="")
for name in target_names:
    print(f"{name:>12}", end="")
print()
for i, name in enumerate(target_names):
    print(f"{name:>10}", end="")
    for j in range(len(target_names)):
        print(f"{cm[i][j]:>12}", end="")
    print()
print()

# Print the F1 Score (weighted average) for overall model evaluation
# Weighted F1 accounts for class imbalance by averaging per-class scores
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1 Score (weighted avg): {f1:.4f}")
print()

# Also print overall accuracy for reference
# Accuracy = percentage of correct predictions out of total test samples
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy:        {accuracy:.4f}")
print()
print("=" * 50)
