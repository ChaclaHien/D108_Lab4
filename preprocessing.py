import numpy as np


def train_test_split(X, y, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    split = int(len(X) * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def normalize(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


if __name__ == "__main__":
    from data_loader import load_data
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train, X_test = normalize(X_train, X_test)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
