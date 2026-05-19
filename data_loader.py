"""
Data loader for Wine Quality dataset.

Priority:
  1. Local CSV files in `data/` directory (UCI format, semicolon-separated)
  2. Download from UCI repository
  3. Fallback: sklearn wine dataset (different, 3-class, for smoke-testing only)

Place winequality-red.csv and/or winequality-white.csv in `data/` for best results.
Download from: https://archive.ics.uci.edu/dataset/186/wine+quality
"""
import numpy as np
import os

DATA_URL_RED = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
DATA_URL_WHITE = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"


def _try_download(url, fpath):
    try:
        import urllib.request
        print(f"Downloading {os.path.basename(fpath)}...")
        urllib.request.urlretrieve(url, fpath)
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def load_data(path="data", wine_type="both", binary=True):
    """
    Load Wine Quality dataset.
    binary=True  → quality >= 6 = 1 (good), else 0 (bad)  [binary classification]
    binary=False → keep original quality scores 3-9        [multi-class]
    """
    os.makedirs(path, exist_ok=True)
    files = {
        "red": ("red.csv", DATA_URL_RED),
        "white": ("white.csv", DATA_URL_WHITE),
    }

    selected = ["red", "white"] if wine_type == "both" else [wine_type]
    datasets = []

    for key in selected:
        fname, url = files[key]
        fpath = os.path.join(path, fname)
        if not os.path.exists(fpath):
            _try_download(url, fpath)

        if os.path.exists(fpath):
            arr = np.genfromtxt(fpath, delimiter=";", skip_header=1)
            datasets.append(arr)
        else:
            print(f"  '{fname}' not found. Skipping.")

    if datasets:
        data = np.vstack(datasets)
        X = data[:, :-1]
        y = data[:, -1].astype(int)
        if binary:
            y = (y >= 6).astype(int)
        print(f"Loaded {len(X)} samples, {X.shape[1]} features.")
        return X, y

    # ── Fallback: sklearn wine (3-class, different dataset) ──────────
    print("WARNING: Using sklearn's wine dataset as fallback (3 classes, 178 samples).")
    print("         For the real Wine Quality dataset, place UCI CSVs in the data/ folder.")
    from sklearn.datasets import load_wine
    data = load_wine()
    X = data.data.astype(float)
    y = data.target.astype(int)
    return X, y


if __name__ == "__main__":
    X, y = load_data()
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    unique, counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip(unique.tolist(), counts.tolist()))}")
