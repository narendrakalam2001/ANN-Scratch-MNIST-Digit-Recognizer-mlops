# ============================================================
# MODEL TUNING — ANN From Scratch: MNIST Digit Recognizer
# ============================================================
#
# Since the network itself is hand-rolled (no sklearn estimator
# interface), hyperparameter search is also hand-rolled: a small
# random search over architecture + learning rate, each trial
# trained for a reduced number of epochs on train/val, ranked by
# validation accuracy. The winning config is then retrained to
# full convergence in training_pipeline.py.
# ============================================================

import numpy as np
import logging
import time

from src.neural_network import NeuralNetworkFromScratch
from src.config import (
    INPUT_DIM,
    NUM_CLASSES,
    HIDDEN_LAYER_CHOICES,
    LR_CHOICES,
    TUNING_TRIALS,
    RANDOM_STATE,
    BATCH_SIZE,
    L2_LAMBDA,
    DROPOUT_KEEP_PROB,
    OPTIMIZER,
    WEIGHT_INIT,
)

logger = logging.getLogger(__name__)

TUNING_EPOCHS = 8  # short budget per trial -- full training happens after selection


def random_search(X_train, y_train_oh, X_val, y_val_oh, n_trials: int = TUNING_TRIALS):
    """
    Randomly samples `n_trials` (hidden_layers, learning_rate) combinations,
    trains each briefly, and returns a leaderboard sorted by val accuracy.
    """
    rng = np.random.RandomState(RANDOM_STATE)
    results = []

    for trial in range(1, n_trials + 1):
        hidden_layers = HIDDEN_LAYER_CHOICES[rng.randint(len(HIDDEN_LAYER_CHOICES))]
        lr = LR_CHOICES[rng.randint(len(LR_CHOICES))]
        layer_dims = [INPUT_DIM] + list(hidden_layers) + [NUM_CLASSES]

        logger.info("Trial %d/%d | hidden=%s | lr=%.4f", trial, n_trials, hidden_layers, lr)

        t0 = time.time()
        net = NeuralNetworkFromScratch(
            layer_dims=layer_dims,
            activation="relu",
            weight_init=WEIGHT_INIT,
            optimizer=OPTIMIZER,
            learning_rate=lr,
            l2_lambda=L2_LAMBDA,
            dropout_keep_prob=DROPOUT_KEEP_PROB,
            random_state=RANDOM_STATE,
        )
        net.fit(
            X_train,
            y_train_oh,
            X_val,
            y_val_oh,
            epochs=TUNING_EPOCHS,
            batch_size=BATCH_SIZE,
            early_stopping_patience=TUNING_EPOCHS,  # no early stop in short trials
            verbose=False,
        )
        val_acc = net.history["val_acc"][-1]
        elapsed = time.time() - t0

        results.append(
            {
                "trial": trial,
                "hidden_layers": hidden_layers,
                "learning_rate": lr,
                "val_acc": round(val_acc, 4),
                "seconds": round(elapsed, 1),
            }
        )
        logger.info("  -> val_acc=%.4f (%.1fs)", val_acc, elapsed)

    results.sort(key=lambda r: r["val_acc"], reverse=True)
    logger.info("Best config: %s", results[0])
    return results


def best_config_from_search(results: list) -> dict:
    best = results[0]
    return {"hidden_layers": best["hidden_layers"], "learning_rate": best["learning_rate"]}
