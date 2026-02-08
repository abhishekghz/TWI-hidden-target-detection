from typing import Sequence


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    return correct / max(len(y_true), 1)


def macro_precision_recall_f1(y_true: Sequence[int], y_pred: Sequence[int]):
    labels = sorted(set(y_true) | set(y_pred))
    precisions = []
    recalls = []
    f1s = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return {
        "precision": sum(precisions) / max(len(precisions), 1),
        "recall": sum(recalls) / max(len(recalls), 1),
        "f1": sum(f1s) / max(len(f1s), 1),
    }
