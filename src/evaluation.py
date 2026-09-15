# ============================================================
# EVALUATION — ANN From Scratch: MNIST Digit Recognizer
# ============================================================

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.metrics import (
    accuracy, confusion_matrix_np, precision_recall_f1_per_class,
    macro_f1, top_k_accuracy, mean_top1_confidence, generalization_gap,
    misclassification_cost_report,
)
from src.config import NUM_CLASSES, MODEL_DIR

logger = logging.getLogger(__name__)


# ============================================================
# FULL EVALUATION REPORT
# ============================================================

def evaluate_network(network, X_train, y_train, X_test, y_test, dataset_label="test"):
    """
    Runs a full evaluation of a trained NeuralNetworkFromScratch and
    returns a metrics dict ready to drop into a model card / challenger
    comparison.
    """
    train_pred = network.predict(X_train)
    train_acc = accuracy(y_train, train_pred)

    test_prob = network.predict_proba(X_test)
    test_pred = np.argmax(test_prob, axis=1)
    test_acc = accuracy(y_test, test_pred)
    test_f1 = macro_f1(y_test, test_pred, NUM_CLASSES)
    test_top2 = top_k_accuracy(y_test, test_prob, k=2)
    confidence = mean_top1_confidence(test_prob)
    gap = generalization_gap(train_acc, test_acc)

    precision, recall, f1 = precision_recall_f1_per_class(y_test, test_pred, NUM_CLASSES)

    metrics = {
        "train_accuracy": round(train_acc, 4),
        f"{dataset_label}_accuracy": round(test_acc, 4),
        f"{dataset_label}_macro_f1": round(test_f1, 4),
        f"{dataset_label}_top2_accuracy": round(test_top2, 4),
        "mean_top1_confidence": round(confidence, 4),
        "generalization_gap": round(gap, 4),
        "per_class_precision": {str(i): round(precision[i], 4) for i in range(NUM_CLASSES)},
        "per_class_recall": {str(i): round(recall[i], 4) for i in range(NUM_CLASSES)},
        "per_class_f1": {str(i): round(f1[i], 4) for i in range(NUM_CLASSES)},
    }

    cost_report = misclassification_cost_report(y_test, test_pred)
    metrics["business_impact"] = cost_report

    logger.info(
        "Eval | train_acc=%.4f  %s_acc=%.4f  macro_f1=%.4f  gap=%.4f",
        train_acc, dataset_label, test_acc, test_f1, gap,
    )

    return metrics, test_pred, test_prob


# ============================================================
# PLOTS
# ============================================================

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix_np(y_true, y_pred, NUM_CLASSES)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted digit")
    plt.ylabel("True digit")
    plt.title("Confusion Matrix — ANN From Scratch (MNIST)")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        logger.info("Saved confusion matrix -> %s", save_path)
    plt.close()


def plot_training_curves(history, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history["train_loss"], label="train_loss")
    if history.get("val_loss"):
        axes[0].plot(history["val_loss"], label="val_loss")
    axes[0].set_title("Loss (cross-entropy)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train_acc")
    if history.get("val_acc"):
        axes[1].plot(history["val_acc"], label="val_acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        logger.info("Saved training curves -> %s", save_path)
    plt.close()


def plot_per_class_f1(metrics, save_path=None):
    """
    FIX (2026-08-19): previously passed string labels like ['0','1',...]
    as the x-axis to sns.barplot. Even though they were strings, matplotlib
    detects they're "parsable as floats" and logs a noisy INFO message via
    matplotlib.category ("Using categorical units to plot a list of
    strings..."). Using plain integer x-positions with explicit tick
    labels avoids the categorical-units code path entirely, so the
    message never fires.
    """
    f1_dict = metrics["per_class_f1"]
    digits = [int(d) for d in f1_dict.keys()]
    scores = [f1_dict[str(d)] for d in digits]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(digits, scores, color="#4C72B0")
    ax.set_xticks(digits)
    ax.set_xticklabels([str(d) for d in digits])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Digit")
    ax.set_ylabel("F1 score")
    ax.set_title("Per-Class F1 — ANN From Scratch (MNIST)")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        logger.info("Saved per-class F1 chart -> %s", save_path)
    plt.close()


def plot_misclassified_samples(X_test, y_true, y_pred, n=16, save_path=None):
    """Grid of misclassified digit images with true/predicted labels — the
    classic 'where does the model get confused' interview visual."""
    wrong_idx = np.where(y_true != y_pred)[0]
    if len(wrong_idx) == 0:
        logger.info("No misclassified samples to plot.")
        return
    sample_idx = np.random.RandomState(42).choice(wrong_idx, size=min(n, len(wrong_idx)), replace=False)

    cols = 4
    rows = int(np.ceil(len(sample_idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.2))
    axes = np.array(axes).reshape(-1)

    for ax, idx in zip(axes, sample_idx):
        img = X_test[idx].reshape(28, 28)
        ax.imshow(img, cmap="gray")
        ax.set_title(f"true={y_true[idx]} pred={y_pred[idx]}", fontsize=9)
        ax.axis("off")
    for ax in axes[len(sample_idx):]:
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        logger.info("Saved misclassified-samples grid -> %s", save_path)
    plt.close()