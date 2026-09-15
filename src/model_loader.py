# ============================================================
# MODEL LOADER + CHALLENGER SYSTEM
# ANN From Scratch: MNIST Digit Recognizer
# ============================================================
#
# CHAMPION-CHALLENGER SYSTEM:
#   Champion   = current production ANN (latest_model.json)
#   Challenger = newly trained ANN (passed in from training_pipeline)
#
#   Promotion logic — challenger promoted ONLY if it beats champion
#   on ALL 3 gates:
#     1. Test accuracy improvement >= MIN_ACCURACY_IMPROVEMENT
#     2. Macro F1                  >= MIN_MACRO_F1_THRESHOLD
#     3. Generalization gap        <= MAX_GENERALIZATION_GAP
#
#   If challenger loses -> champion stays, challenger archived.
#   Full comparison saved to ann_models/challenger_log.json
# ============================================================

import os
import json
import logging
import time

from src.config import MODEL_DIR, MIN_ACCURACY_IMPROVEMENT, MIN_MACRO_F1_THRESHOLD, MAX_GENERALIZATION_GAP
from src.neural_network import NeuralNetworkFromScratch

logger = logging.getLogger(__name__)

CHALLENGER_LOG = os.path.join(MODEL_DIR, "challenger_log.json")


# ============================================================
# LOAD LATEST (CHAMPION) MODEL
# ============================================================


def load_latest_model():
    """
    Reads ann_models/latest_model.json -> loads .npz weights + config.
    Returns: (network, registry_dict)
    """
    registry_path = os.path.join(MODEL_DIR, "latest_model.json")

    if not os.path.exists(registry_path):
        raise FileNotFoundError(
            f"Model registry not found at {registry_path}. Run scripts/train_model.py first."
        )

    with open(registry_path) as f:
        registry = json.load(f)

    path_prefix = os.path.join(MODEL_DIR, registry["model_prefix"])
    if not os.path.exists(f"{path_prefix}.npz"):
        raise FileNotFoundError(f"Model weights not found: {path_prefix}.npz")

    network = NeuralNetworkFromScratch.load(path_prefix)
    logger.info("Champion model loaded: %s", path_prefix)

    return network, registry


# ============================================================
# LOAD CHAMPION METRICS FROM MODEL CARD
# ============================================================


def _load_champion_metrics() -> dict:
    registry_path = os.path.join(MODEL_DIR, "latest_model.json")
    if not os.path.exists(registry_path):
        return {}

    with open(registry_path) as f:
        registry = json.load(f)

    card_path = registry.get("model_card_path", "")
    if not card_path or not os.path.exists(card_path):
        logger.warning("Champion model card not found: %s", card_path)
        return {}

    with open(card_path) as f:
        card = json.load(f)

    metrics = card.get("metrics", {})
    return {
        "model_name": card.get("model_details", {}).get("name", "unknown"),
        "accuracy": float(metrics.get("test_accuracy", 0)),
        "macro_f1": float(metrics.get("test_macro_f1", 0)),
    }


# ============================================================
# CHALLENGER COMPARISON — CORE LOGIC
# ============================================================


def run_challenger_comparison(
    challenger_name: str,
    challenger_accuracy: float,
    challenger_macro_f1: float,
    challenger_gap: float,
    challenger_prefix: str,  # path prefix, e.g. ann_models/ann_v3
    challenger_card_path: str = "",
) -> dict:
    os.makedirs(MODEL_DIR, exist_ok=True)

    champion = _load_champion_metrics()

    # ── Safety net (2026-08-20): if the "champion" we just loaded has the
    # SAME name as the challenger, the paths collided and we're about to
    # compare the challenger against its own just-written card -- this
    # should never happen now that training_pipeline.py's versioning is
    # registry-aware, but if it ever does (e.g. manual file edits, a
    # renamed run), treat it as "no valid champion" rather than silently
    # producing a false REJECTED with identical metrics on both sides. ──
    if champion and champion.get("model_name") == challenger_name:
        logger.error(
            "Champion name collides with challenger name ('%s') -- the champion's "
            "card was likely just overwritten by this run's own save_model_card() call. "
            "Treating as no valid champion to avoid comparing the challenger to itself.",
            challenger_name,
        )
        champion = {}

    if not champion:
        logger.info("No champion found -- challenger auto-promoted as first model")
        _update_registry(challenger_prefix, challenger_card_path)
        result = {
            "decision": "PROMOTED",
            "reason": "No existing champion -- first model auto-promoted",
            "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "challenger_name": challenger_name,
            "challenger_accuracy": round(challenger_accuracy, 4),
            "challenger_macro_f1": round(challenger_macro_f1, 4),
            "champion_name": None,
            "champion_accuracy": None,
        }
        _save_challenger_log(result)
        return result

    champion_acc = champion.get("accuracy", 0.0)
    champion_f1 = champion.get("macro_f1", 0.0)
    champion_name = champion.get("model_name", "unknown")

    logger.info("=" * 55)
    logger.info("CHAMPION vs CHALLENGER")
    logger.info("  Champion  : %-20s acc=%.4f  macro_f1=%.4f", champion_name, champion_acc, champion_f1)
    logger.info(
        "  Challenger: %-20s acc=%.4f  macro_f1=%.4f",
        challenger_name,
        challenger_accuracy,
        challenger_macro_f1,
    )
    logger.info("=" * 55)

    gate1_acc_improvement = (challenger_accuracy - champion_acc) >= MIN_ACCURACY_IMPROVEMENT
    gate2_macro_f1 = challenger_macro_f1 >= MIN_MACRO_F1_THRESHOLD
    gate3_gap = challenger_gap <= MAX_GENERALIZATION_GAP

    gates_passed = gate1_acc_improvement and gate2_macro_f1 and gate3_gap

    if gates_passed:
        decision = "PROMOTED"
        reason = (
            f"Challenger beats champion: accuracy {champion_acc:.4f} -> {challenger_accuracy:.4f} "
            f"(+{challenger_accuracy - champion_acc:.4f})"
        )
        logger.info("CHALLENGER PROMOTED -> new champion: %s", challenger_name)
        _update_registry(challenger_prefix, challenger_card_path)
    else:
        decision = "REJECTED"
        failed = []
        if not gate1_acc_improvement:
            failed.append(
                f"accuracy improvement {challenger_accuracy - champion_acc:+.4f} < {MIN_ACCURACY_IMPROVEMENT}"
            )
        if not gate2_macro_f1:
            failed.append(f"macro F1 {challenger_macro_f1:.4f} < {MIN_MACRO_F1_THRESHOLD}")
        if not gate3_gap:
            failed.append(f"generalization gap {challenger_gap:.4f} > {MAX_GENERALIZATION_GAP}")
        reason = "Gates failed: " + " | ".join(failed)
        logger.info("CHALLENGER REJECTED -- champion '%s' retained", champion_name)
        logger.info("   Reason: %s", reason)

    result = {
        "decision": decision,
        "reason": reason,
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "challenger_name": challenger_name,
        "challenger_accuracy": round(challenger_accuracy, 4),
        "challenger_macro_f1": round(challenger_macro_f1, 4),
        "challenger_gap": round(challenger_gap, 4),
        "champion_name": champion_name,
        "champion_accuracy": round(champion_acc, 4),
        "champion_macro_f1": round(champion_f1, 4),
        "gates": {
            "accuracy_improvement_passed": gate1_acc_improvement,
            "macro_f1_passed": gate2_macro_f1,
            "gap_passed": gate3_gap,
        },
    }
    _save_challenger_log(result)
    return result


# ============================================================
# HELPERS
# ============================================================


def _update_registry(model_prefix: str, model_card_path: str = None):
    registry = {
        "model_prefix": os.path.basename(model_prefix),
        "model_card_path": model_card_path or "",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(MODEL_DIR, "latest_model.json"), "w") as f:
        json.dump(registry, f, indent=2)
    logger.info("Registry updated -> %s", registry["model_prefix"])


def _save_challenger_log(result: dict):
    history = []
    if os.path.exists(CHALLENGER_LOG):
        try:
            with open(CHALLENGER_LOG) as f:
                history = json.load(f)
        except Exception:
            history = []

    history.append(result)
    with open(CHALLENGER_LOG, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Challenger log saved -> %s", CHALLENGER_LOG)