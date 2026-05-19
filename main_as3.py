"""
Assignment 3: Decision Tree and Random Forest using scikit-learn.
Also runs AS1 & AS2 and compares all 4 models side by side with a chart.
"""
from data_loader import load_data
from preprocessing import train_test_split, normalize
from metrics import f1_score, report
from sklearn_models import run as run_sklearn
from decision_tree_numpy import DecisionTreeNumpy
from random_forest_numpy import RandomForestNumpy
from visualization import plot_f1_comparison

from sklearn.metrics import f1_score as sk_f1, accuracy_score


def eval_sklearn(model, X_test, y_test, label):
    preds = model.predict(X_test)
    report(y_test, preds, label)
    return {
        "f1_macro": sk_f1(y_test, preds, average="macro"),
        "f1_weighted": sk_f1(y_test, preds, average="weighted"),
    }


def main():
    print("=== Assignment 3: All Models Comparison ===")

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)
    X_train, X_test = normalize(X_train, X_test)

    results = {}

    # ── NumPy DT ─────────────────────────────────────────────────────
    dt_np = DecisionTreeNumpy(max_depth=8, min_samples_split=5)
    dt_np.fit(X_train, y_train)
    preds = dt_np.predict(X_test)
    results["DT (NumPy)"] = {
        "f1_macro": f1_score(y_test, preds, "macro"),
        "f1_weighted": f1_score(y_test, preds, "weighted"),
    }
    report(y_test, preds, "DT (NumPy)")

    # ── NumPy RF ─────────────────────────────────────────────────────
    rf_np = RandomForestNumpy(n_estimators=50, max_depth=8, min_samples_split=5, seed=42)
    print("\nTraining NumPy Random Forest...")
    rf_np.fit(X_train, y_train)
    preds = rf_np.predict(X_test)
    results["RF (NumPy)"] = {
        "f1_macro": f1_score(y_test, preds, "macro"),
        "f1_weighted": f1_score(y_test, preds, "weighted"),
    }
    report(y_test, preds, "RF (NumPy)")

    # ── sklearn ─────────────────────────────────────────────────────
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier

    dt_sk = DecisionTreeClassifier(max_depth=8, random_state=42)
    dt_sk.fit(X_train, y_train)
    results["DT (sklearn)"] = eval_sklearn(dt_sk, X_test, y_test, "DT (sklearn)")

    rf_sk = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf_sk.fit(X_train, y_train)
    results["RF (sklearn)"] = eval_sklearn(rf_sk, X_test, y_test, "RF (sklearn)")

    # ── Visualization ────────────────────────────────────────────────
    plot_f1_comparison(results, save_path="f1_comparison.png")

    print("\n=== Summary ===")
    for name, scores in results.items():
        print(f"  {name:<15} F1 Macro={scores['f1_macro']:.4f}  F1 Weighted={scores['f1_weighted']:.4f}")


if __name__ == "__main__":
    main()
