from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report


def evaluate(model, X_test, y_test, label="Model"):
    preds = model.predict(X_test)
    print(f"\n{'='*40}")
    print(f"  {label}")
    print(f"{'='*40}")
    print(f"  Accuracy : {accuracy_score(y_test, preds):.4f}")
    print(f"  F1 Macro : {f1_score(y_test, preds, average='macro'):.4f}")
    print(f"  F1 Weight: {f1_score(y_test, preds, average='weighted'):.4f}")
    print(classification_report(y_test, preds, target_names=["Bad", "Good"]))


def run(X_train, X_test, y_train, y_test):
    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=8, random_state=42)
    dt.fit(X_train, y_train)
    evaluate(dt, X_test, y_test, "Decision Tree (scikit-learn)")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    evaluate(rf, X_test, y_test, "Random Forest (scikit-learn)")

    return dt, rf


if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import train_test_split, normalize

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train, X_test = normalize(X_train, X_test)
    run(X_train, X_test, y_train, y_test)
