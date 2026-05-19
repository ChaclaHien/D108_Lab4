import numpy as np
from decision_tree_numpy import DecisionTreeNumpy


class RandomForestNumpy:
    def __init__(self, n_estimators=50, max_depth=10, min_samples_split=2,
                 n_features="sqrt", seed=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features  # "sqrt", "log2", or int
        self.seed = seed
        self.trees = []

    def _resolve_n_features(self, total_features):
        if self.n_features == "sqrt":
            return max(1, int(np.sqrt(total_features)))
        elif self.n_features == "log2":
            return max(1, int(np.log2(total_features)))
        elif isinstance(self.n_features, int):
            return self.n_features
        return total_features

    def fit(self, X, y):
        np.random.seed(self.seed)
        n_samples, n_feat = X.shape
        n_feat_use = self._resolve_n_features(n_feat)
        self.trees = []

        for _ in range(self.n_estimators):
            # Bootstrap sampling
            idx = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot, y_boot = X[idx], y[idx]

            tree = DecisionTreeNumpy(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=n_feat_use,
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

        return self

    def predict(self, X):
        # Shape: (n_estimators, n_samples)
        all_preds = np.stack([tree.predict(X) for tree in self.trees], axis=0)
        # Majority vote
        return np.apply_along_axis(
            lambda col: np.bincount(col).argmax(), axis=0, arr=all_preds
        )


if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import train_test_split, normalize
    from metrics import report

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train, X_test = normalize(X_train, X_test)

    model = RandomForestNumpy(n_estimators=50, max_depth=8)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    report(y_test, preds, "Random Forest (NumPy)")
