"""
Assignment 2: Random Forest implemented from scratch with NumPy.
Evaluated on the Wine Quality dataset using F1 score.
"""
from data_loader import load_data
from preprocessing import train_test_split, normalize
from random_forest_numpy import RandomForestNumpy
from metrics import f1_score, report


def main():
    print("=== Assignment 2: Random Forest (NumPy) ===")

    # 1. Load data
    X, y = load_data()

    # 2. Split & normalize
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)
    X_train, X_test = normalize(X_train, X_test)

    # 3. Train
    model = RandomForestNumpy(n_estimators=50, max_depth=8, min_samples_split=5, seed=42)
    print("Training Random Forest... (may take ~30s)")
    model.fit(X_train, y_train)

    # 4. Evaluate
    preds = model.predict(X_test)
    report(y_test, preds, "Random Forest (NumPy) — AS2")

    return f1_score(y_test, preds, average="macro")


if __name__ == "__main__":
    main()
