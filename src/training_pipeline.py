# ============================================================
# TRAINING PIPELINE — ANN From Scratch: MNIST Digit Recognizer
# ============================================================
#
# End-to-end orchestration:
#   1. Load + validate raw MNIST CSV
#   2. Stratified train/val/test split
#   3. Leakage check across splits
#   4. Preprocess (scale + one-hot) — fit on train only
#   5. Hyperparameter search (short trials) -> pick best config
#   6. Train final network to convergence on best config
#   7. Evaluate on held-out test set
#   8. Build model card (Google Model Cards standard)
#   9. Run champion-challenger comparison -> promote or reject
#  10. Save plots + logs
#
# FIX (2026-08-19): model_version now auto-increments (v1, v2, v3...)
# instead of always defaulting to "v1". Previously every run reused the
# same filename for weights + model card, which meant a new challenger's
# save_model_card() call silently overwrote the CURRENT CHAMPION's model
# card on disk before run_challenger_comparison() ever read it — so the
# comparison always compared the challenger against itself (identical
# metrics, "accuracy improvement +0.0000") and rejected every run, while
# also destroying the previous champion's saved card/weights. Auto-
# versioning gives every run a distinct file path, so old champion
# artifacts are never touched unless/until a new model is promoted.
# ============================================================

import os
import re
import json
import logging
import numpy as np

from src.config import (
    DATA_DIR, MODEL_DIR, VAL_SIZE, TEST_SIZE, RANDOM_STATE,
    INPUT_DIM, NUM_CLASSES, EPOCHS, BATCH_SIZE, LR_DECAY,
    EARLY_STOPPING_PATIENCE, L2_LAMBDA, DROPOUT_KEEP_PROB,
    OPTIMIZER, WEIGHT_INIT, TUNING_TRIALS,
)
from src.data_loader import load_train_data, class_distribution
from src.preprocessing import train_val_test_split, preprocess_pipeline
from src.leakage_check import check_split_leakage, check_label_consistency
from src.model_tuning import random_search, best_config_from_search
from src.neural_network import NeuralNetworkFromScratch
from src.evaluation import (
    evaluate_network, plot_confusion_matrix, plot_training_curves,
    plot_per_class_f1, plot_misclassified_samples,
)
from src.model_card import build_model_card, save_model_card
from src.model_loader import run_challenger_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_BASE_NAME = "ANN_Scratch"


def _next_model_version(model_dir: str = MODEL_DIR, base_name: str = MODEL_BASE_NAME) -> str:
    """
    Scans model_dir for existing '{base_name}_v<N>.npz' files AND checks
    the registry's (latest_model.json) recorded champion name, and
    returns the next version string ('v1' if none exist yet).

    FIX (2026-08-20): scanning disk files alone is not enough. If the
    registry still records a champion (e.g. "ANN_Scratch_v1") but that
    model's .npz/card files are missing on disk (deleted, never copied
    to a new machine, etc), a disk-only scan would reuse "v1" again --
    colliding with the path the registry still considers the champion's
    card, and reproducing the exact overwrite-then-compare-to-self bug
    this function exists to prevent. We now also parse the registry's
    model_prefix and treat that version number as reserved.
    """
    versions = []

    if os.path.isdir(model_dir):
        pattern = re.compile(rf"^{re.escape(base_name)}_v(\d+)\.npz$")
        for fname in os.listdir(model_dir):
            m = pattern.match(fname)
            if m:
                versions.append(int(m.group(1)))

    registry_path = os.path.join(model_dir, "latest_model.json")
    if os.path.exists(registry_path):
        try:
            with open(registry_path) as f:
                registry = json.load(f)
            reg_prefix = registry.get("model_prefix", "") or ""
            m = re.match(rf"^{re.escape(base_name)}_v(\d+)$", reg_prefix)
            if m:
                versions.append(int(m.group(1)))
        except Exception as e:
            logger.warning("Could not parse existing registry for version dedup: %s", e)

    next_v = (max(versions) + 1) if versions else 1
    return f"v{next_v}"


def run_training_pipeline(
    data_path: str = None,
    run_tuning: bool = True,
    epochs: int = EPOCHS,
    model_version: str = None,
):
    logger.info("=" * 60)
    logger.info("STARTING TRAINING PIPELINE — ANN From Scratch (MNIST)")
    logger.info("=" * 60)

    # ── Auto-version if not explicitly provided ────────────
    if model_version is None:
        model_version = _next_model_version()
    logger.info("This run will be saved as model_version=%s", model_version)

    # ── 1. Load ──────────────────────────────────────────────
    data_path = data_path or os.path.join(DATA_DIR, "train.csv")
    X, y = load_train_data(DATA_DIR, os.path.basename(data_path))
    logger.info("Class distribution:\n%s", class_distribution(y).to_string())

    # ── 2. Split ─────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, val_size=VAL_SIZE, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # ── 3. Leakage check (BEFORE any fitting) ───────────────
    check_split_leakage(X_train, X_val, X_test)
    check_label_consistency(y_train, y_val, y_test, NUM_CLASSES)

    # ── 4. Preprocess ────────────────────────────────────────
    data = preprocess_pipeline(X_train, X_val, X_test, y_train, y_val, y_test)

    # ── 5. Hyperparameter search ────────────────────────────
    if run_tuning:
        search_results = random_search(
            data["X_train"], data["y_train_oh"], data["X_val"], data["y_val_oh"],
            n_trials=TUNING_TRIALS,
        )
        best_cfg = best_config_from_search(search_results)
        hidden_layers, learning_rate = best_cfg["hidden_layers"], best_cfg["learning_rate"]
    else:
        from src.config import HIDDEN_LAYERS, LEARNING_RATE
        hidden_layers, learning_rate = HIDDEN_LAYERS, LEARNING_RATE

    logger.info("Final architecture: hidden_layers=%s | lr=%.4f", hidden_layers, learning_rate)

    # ── 6. Train final network to convergence ───────────────
    layer_dims = [INPUT_DIM] + list(hidden_layers) + [NUM_CLASSES]
    network = NeuralNetworkFromScratch(
        layer_dims=layer_dims,
        activation="relu",
        weight_init=WEIGHT_INIT,
        optimizer=OPTIMIZER,
        learning_rate=learning_rate,
        l2_lambda=L2_LAMBDA,
        dropout_keep_prob=DROPOUT_KEEP_PROB,
        random_state=RANDOM_STATE,
    )
    history = network.fit(
        data["X_train"], data["y_train_oh"],
        data["X_val"], data["y_val_oh"],
        epochs=epochs, batch_size=BATCH_SIZE, lr_decay=LR_DECAY,
        early_stopping_patience=EARLY_STOPPING_PATIENCE, verbose=True, log_every=5,
    )

    # ── 7. Evaluate on held-out test set ────────────────────
    metrics, test_pred, test_prob = evaluate_network(
        network, data["X_train"], data["y_train"], data["X_test"], data["y_test"]
    )

    # ── 8. Plots ─────────────────────────────────────────────
    plot_confusion_matrix(data["y_test"], test_pred, "docs/plots/confusion_matrix.png")
    plot_training_curves(history, "docs/plots/training_curves.png")
    plot_per_class_f1(metrics, "docs/plots/per_class_f1.png")
    plot_misclassified_samples(data["X_test"], data["y_test"], test_pred, save_path="docs/plots/misclassified_samples.png")

    # ── 9. Save network weights (unique path per version) ──
    model_name = f"{MODEL_BASE_NAME}_{model_version}"
    model_prefix = os.path.join(MODEL_DIR, model_name)
    network.save(model_prefix)
    logger.info("Model weights saved -> %s.npz", model_prefix)

    # ── 10. Model card (unique path per version) ────────────
    card = build_model_card(
        model_name=model_name,
        version=model_version,
        architecture_config={
            "layer_dims": layer_dims,
            "activation": "relu",
            "weight_init": WEIGHT_INIT,
            "optimizer": OPTIMIZER,
        },
        metrics=metrics,
        thresholds={"decision_rule": "argmax(softmax_output)"},
        training_config={
            "epochs_run": len(history["train_loss"]),
            "batch_size": BATCH_SIZE,
            "learning_rate_initial": learning_rate,
            "lr_decay": LR_DECAY,
            "l2_lambda": L2_LAMBDA,
            "dropout_keep_prob": DROPOUT_KEEP_PROB,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        },
        dataset_info={
            "source": "Kaggle Digit Recognizer (MNIST CSV format)",
            "n_train": len(data["X_train"]),
            "n_val": len(data["X_val"]),
            "n_test": len(data["X_test"]),
            "input_dim": INPUT_DIM,
            "num_classes": NUM_CLASSES,
        },
    )
    card_path = os.path.join(MODEL_DIR, f"model_card_{model_name}.json")
    save_model_card(card, card_path)

    # ── 11. Champion-Challenger comparison ──────────────────
    # At this point model_prefix / card_path are GUARANTEED distinct from
    # whatever the current champion's files are (different version number),
    # so run_challenger_comparison() reads the champion's real prior metrics.
    result = run_challenger_comparison(
        challenger_name=model_name,
        challenger_accuracy=metrics["test_accuracy"],
        challenger_macro_f1=metrics["test_macro_f1"],
        challenger_gap=metrics["generalization_gap"],
        challenger_prefix=model_prefix,
        challenger_card_path=card_path,
    )

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE | decision=%s | test_acc=%.4f | macro_f1=%.4f",
                result["decision"], metrics["test_accuracy"], metrics["test_macro_f1"])
    logger.info("=" * 60)

    return {"network": network, "metrics": metrics, "challenger_result": result, "model_prefix": model_prefix}


if __name__ == "__main__":
    run_training_pipeline()