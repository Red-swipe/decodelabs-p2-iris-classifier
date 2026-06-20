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
