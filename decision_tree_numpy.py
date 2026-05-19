import numpy as np


class Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTreeNumpy:
    def __init__(self, max_depth=10, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features  # used by Random Forest for feature subsampling
        self.root = None

    # ── Gini impurity ──────────────────────────────────────────────────
    @staticmethod
    def _gini(y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1.0 - float(np.sum(probs ** 2))

    def _best_split(self, X, y, features):
        best_gain, best_feat, best_thresh = -1, None, None
        parent_gini = self._gini(y)
        n = len(y)

        for feat in features:
            thresholds = np.unique(X[:, feat])
            thresholds = (thresholds[:-1] + thresholds[1:]) / 2  # midpoints

            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                n_l, n_r = left_mask.sum(), n - left_mask.sum()
                if n_l == 0 or n_r == 0:
                    continue

                gain = parent_gini - (
                    n_l / n * self._gini(y[left_mask]) +
                    n_r / n * self._gini(y[~left_mask])
                )
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, thresh

        return best_feat, best_thresh

    def _build(self, X, y, depth):
        # Stopping criteria
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return Node(value=int(np.bincount(y).argmax()))

        # Feature subsampling (supports Random Forest usage)
        n_feat = self.n_features or X.shape[1]
        n_feat = min(n_feat, X.shape[1])
        features = np.random.choice(X.shape[1], size=n_feat, replace=False)

        feat, thresh = self._best_split(X, y, features)
        if feat is None:
            return Node(value=int(np.bincount(y).argmax()))

        left_mask = X[:, feat] <= thresh
        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return Node(feature=feat, threshold=thresh, left=left, right=right)

    def fit(self, X, y):
        self.root = self._build(X, y, depth=0)
        return self

    def _predict_one(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])


if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import train_test_split, normalize
    from metrics import report

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train, X_test = normalize(X_train, X_test)

    model = DecisionTreeNumpy(max_depth=8)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    report(y_test, preds, "Decision Tree (NumPy)")
