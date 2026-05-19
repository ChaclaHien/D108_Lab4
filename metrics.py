import numpy as np


def confusion_matrix_binary(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    return tp, fp, fn, tn


def f1_score(y_true, y_pred, average="macro"):
    classes = np.unique(y_true)
    f1s = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1s.append(f1)

    if average == "macro":
        return float(np.mean(f1s))
    elif average == "weighted":
        weights = [np.sum(y_true == c) for c in classes]
        return float(np.average(f1s, weights=weights))
    else:
        return f1s


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def report(y_true, y_pred, label="Model"):
    print(f"\n{'='*40}")
    print(f"  {label}")
    print(f"{'='*40}")
    print(f"  Accuracy : {accuracy(y_true, y_pred):.4f}")
    print(f"  F1 Macro : {f1_score(y_true, y_pred, 'macro'):.4f}")
    print(f"  F1 Weight: {f1_score(y_true, y_pred, 'weighted'):.4f}")
