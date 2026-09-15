# ============================================================
# PREPROCESSING — ANN From Scratch: MNIST Digit Recognizer
# ============================================================
#
# Everything here is hand-written NumPy (no sklearn Scaler/Encoder)
# to stay consistent with the "from scratch" philosophy of the
# project — the model AND the data prep are framework-free.
# ============================================================

import numpy as np
import logging

from src.config import PIXEL_MAX, NUM_CLASSES, RANDOM_STATE

logger = logging.getLogger(__name__)


# ============================================================
# PIXEL SCALER — fit on train, applied identically to val/test
# ============================================================


class PixelScaler:
    """
    Min-max scales raw [0, 255] pixel intensities to [0, 1].
    Stateless in practice (fixed range), but implemented with
    fit/transform so it plugs into the same pattern as a fitted
    sklearn transformer — and so it can be swapped for per-dataset
    min/max if ever trained on a non-8-bit image source.
    """

    def __init__(self, feature_max: float = PIXEL_MAX):
        self.feature_max = feature_max
        self.fitted_ = False

    def fit(self, X: np.ndarray):
        observed_max = float(X.max()) if X.size else self.feature_max
        # Guard against a train slice that happens not to contain a 255 pixel
        self.scale_ = max(observed_max, self.feature_max)
        self.fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("PixelScaler must be fit() before transform().")
        return np.clip(X, 0, self.scale_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ============================================================
# ONE-HOT ENCODING — hand-rolled, no sklearn OneHotEncoder
# ============================================================


def one_hot_encode(y: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    m = y.shape[0]
    one_hot = np.zeros((m, num_classes), dtype=np.float64)
    one_hot[np.arange(m), y] = 1.0
    return one_hot


def one_hot_decode(y_onehot: np.ndarray) -> np.ndarray:
    return np.argmax(y_onehot, axis=1)


# ============================================================
# TRAIN / VAL / TEST SPLIT — stratified, hand-rolled
# ============================================================


def stratified_split(X, y, test_size: float, random_state: int = RANDOM_STATE):
    """
    Stratified split implemented without sklearn.train_test_split so the
    class-balance guarantee is auditable: for every digit class, the same
    proportion of samples goes to each split.
    """
    rng = np.random.RandomState(random_state)
    train_idx, test_idx = [], []

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_test = int(round(len(cls_idx) * test_size))
        test_idx.extend(cls_idx[:n_test])
        train_idx.extend(cls_idx[n_test:])

    train_idx = rng.permutation(train_idx)
    test_idx = rng.permutation(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def train_val_test_split(X, y, val_size: float, test_size: float, random_state: int = RANDOM_STATE):
    """Two-stage stratified split -> (X_train, X_val, X_test, y_train, y_val, y_test)."""
    X_train_full, X_test, y_train_full, y_test = stratified_split(
        X, y, test_size=test_size, random_state=random_state
    )
    # val_size is expressed relative to the ORIGINAL dataset; rescale relative to remaining data
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = stratified_split(
        X_train_full, y_train_full, test_size=relative_val_size, random_state=random_state
    )

    logger.info(
        "Split sizes | train=%d  val=%d  test=%d",
        len(X_train),
        len(X_val),
        len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# FULL PREPROCESSING PIPELINE
# ============================================================


def preprocess_pipeline(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Fits PixelScaler on TRAIN ONLY (leakage-safe), applies to val/test,
    and one-hot encodes all labels. Returns everything training_pipeline
    needs in one call.
    """
    scaler = PixelScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    y_train_oh = one_hot_encode(y_train)
    y_val_oh = one_hot_encode(y_val)
    y_test_oh = one_hot_encode(y_test)

    return {
        "X_train": X_train_s,
        "X_val": X_val_s,
        "X_test": X_test_s,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "y_train_oh": y_train_oh,
        "y_val_oh": y_val_oh,
        "y_test_oh": y_test_oh,
        "scaler": scaler,
    }
