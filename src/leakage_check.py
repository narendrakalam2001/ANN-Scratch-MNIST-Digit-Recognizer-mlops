# ============================================================
# LEAKAGE CHECK — ANN From Scratch: MNIST Digit Recognizer
# ============================================================
#
# For image data, "leakage" means the same (or near-identical)
# image appearing in both train and test splits — common in MNIST-
# style datasets due to duplicate scans. We hash each image's raw
# pixel vector and check for exact-duplicate hashes across splits
# BEFORE training. Must be run before training_pipeline proceeds.
# ============================================================

import hashlib
import numpy as np
import logging

logger = logging.getLogger(__name__)


def _hash_images(X: np.ndarray) -> set:
    """Hashes each row's raw pixel bytes -> set of hex digests."""
    hashes = set()
    for row in X:
        h = hashlib.md5(row.astype(np.uint8).tobytes()).hexdigest()
        hashes.add(h)
    return hashes


def check_split_leakage(X_train, X_val, X_test) -> dict:
    """
    Checks for exact duplicate images across train/val/test splits.
    Raises ValueError if any leakage is found — training should not
    proceed on a compromised split.
    """
    train_hashes = _hash_images(X_train)
    val_hashes = _hash_images(X_val)
    test_hashes = _hash_images(X_test)

    train_val_overlap = train_hashes & val_hashes
    train_test_overlap = train_hashes & test_hashes
    val_test_overlap = val_hashes & test_hashes

    report = {
        "train_val_overlap": len(train_val_overlap),
        "train_test_overlap": len(train_test_overlap),
        "val_test_overlap": len(val_test_overlap),
        "n_train_unique": len(train_hashes),
        "n_val_unique": len(val_hashes),
        "n_test_unique": len(test_hashes),
    }

    total_overlap = report["train_val_overlap"] + report["train_test_overlap"] + report["val_test_overlap"]

    if total_overlap > 0:
        logger.error("LEAKAGE DETECTED: %s", report)
        raise ValueError(
            f"Data leakage detected across splits: {total_overlap} duplicate image(s) "
            f"found between train/val/test. Report: {report}"
        )

    logger.info("Leakage check passed — no duplicate images across train/val/test. %s", report)
    return report


def check_label_consistency(y_train, y_val, y_test, num_classes: int = 10) -> dict:
    """Sanity check: every split should contain examples of (most) digit classes,
    otherwise stratification failed silently upstream."""
    report = {}
    for name, y in (("train", y_train), ("val", y_val), ("test", y_test)):
        present = set(np.unique(y).tolist())
        missing = set(range(num_classes)) - present
        report[name] = {"classes_present": len(present), "classes_missing": sorted(missing)}
        if missing:
            logger.warning("Split '%s' is missing digit classes: %s", name, sorted(missing))
    return report
