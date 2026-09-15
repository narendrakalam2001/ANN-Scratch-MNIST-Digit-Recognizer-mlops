# ============================================================
# METRICS — ANN From Scratch: MNIST Digit Recognizer
# ============================================================

import numpy as np
import logging

logger = logging.getLogger(__name__)


# ============================================================
# CORE CLASSIFICATION METRICS (multiclass, hand-rolled + sklearn cross-check)
# ============================================================


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def precision_recall_f1_per_class(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10):
    cm = confusion_matrix_np(y_true, y_pred, num_classes)
    precision, recall, f1 = np.zeros(num_classes), np.zeros(num_classes), np.zeros(num_classes)

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[c] = (
            2 * precision[c] * recall[c] / (precision[c] + recall[c])
            if (precision[c] + recall[c]) > 0
            else 0.0
        )

    return precision, recall, f1


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10) -> float:
    _, _, f1 = precision_recall_f1_per_class(y_true, y_pred, num_classes)
    return float(np.mean(f1))


def top_k_accuracy(y_true: np.ndarray, y_prob: np.ndarray, k: int = 2) -> float:
    """Fraction of samples where the true label is among the top-k predicted classes."""
    top_k_preds = np.argsort(-y_prob, axis=1)[:, :k]
    hits = [y_true[i] in top_k_preds[i] for i in range(len(y_true))]
    return float(np.mean(hits))


def mean_top1_confidence(y_prob: np.ndarray) -> float:
    """Average softmax probability assigned to the predicted class — a
    confidence signal used for low-confidence alerting in monitoring."""
    return float(np.mean(np.max(y_prob, axis=1)))


# ============================================================
# PSI — Population Stability Index (on mean pixel-intensity buckets)
# ============================================================


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Population Stability Index — measures distribution shift between a
    reference distribution (training data) and a new/incoming distribution
    (production traffic), used here on a per-image summary statistic
    (mean pixel intensity per image) rather than raw pixels, since PSI on
    784 raw pixel columns individually would be both noisy and expensive.

    PSI < 0.10        -> stable, no action
    0.10 <= PSI < 0.20 -> moderate drift, monitor
    PSI >= 0.20        -> major drift, investigate / retrain

    Bin edges are derived from `expected` ONLY, then reused for `actual` —
    computing them independently is a common bug that makes PSI ~0 always.
    """
    try:
        expected = np.asarray(expected, dtype=float)
        actual = np.asarray(actual, dtype=float)

        quantiles = np.linspace(0, 100, buckets + 1)
        bin_edges = np.unique(np.percentile(expected, quantiles))
        if len(bin_edges) < 2:
            return 0.0

        bin_edges[0] = min(bin_edges[0], actual.min()) - 1e-9
        bin_edges[-1] = max(bin_edges[-1], actual.max()) + 1e-9

        expected_counts, _ = np.histogram(expected, bins=bin_edges)
        actual_counts, _ = np.histogram(actual, bins=bin_edges)

        expected_pct = expected_counts / max(expected_counts.sum(), 1)
        actual_pct = actual_counts / max(actual_counts.sum(), 1)

        eps = 1e-6
        expected_pct = np.where(expected_pct == 0, eps, expected_pct)
        actual_pct = np.where(actual_pct == 0, eps, actual_pct)

        psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi_value)

    except Exception as e:
        logger.warning("PSI computation failed: %s", e)
        return 0.0


# ============================================================
# GENERALIZATION GAP
# ============================================================


def generalization_gap(train_acc: float, test_acc: float) -> float:
    return float(train_acc - test_acc)


# ============================================================
# COST-SENSITIVE / BUSINESS-STYLE EVALUATION
# ============================================================


def misclassification_cost_report(y_true, y_pred, cost_per_error: float = 1.0, volume_per_day: int = 100_000):
    """
    Frames raw error rate as a business number: at the given daily
    prediction volume, how many wrong digit reads would this model
    produce and what's the notional cost (e.g. manual-review cost per
    misread in a cheque/form-digitization pipeline)?
    """
    error_rate = 1.0 - accuracy(y_true, y_pred)
    daily_errors = error_rate * volume_per_day
    daily_cost = daily_errors * cost_per_error

    return {
        "error_rate": round(error_rate, 4),
        "est_daily_errors": round(daily_errors, 1),
        "est_daily_cost": round(daily_cost, 2),
        "volume_per_day": volume_per_day,
    }
