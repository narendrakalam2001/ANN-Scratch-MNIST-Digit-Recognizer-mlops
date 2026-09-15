# ============================================================
# CONFIGURATION — ANN From Scratch: MNIST Digit Recognizer
# ============================================================

import os

# ── Reproducibility ──────────────────────────────────────────
RANDOM_STATE = 42

# ── Dataset ───────────────────────────────────────────────────
NUM_CLASSES = 10
IMG_SIZE = 28
INPUT_DIM = IMG_SIZE * IMG_SIZE  # 784
PIXEL_MAX = 255.0

# ── Train / val / test split ────────────────────────────────
VAL_SIZE = 0.10  # carved out of training data
TEST_SIZE = 0.10  # held out, never seen during tuning

# ── Network architecture (default / champion topology) ──────
HIDDEN_LAYERS = [256, 128, 64]
ACTIVATION_HIDDEN = "relu"  # relu | leaky_relu | tanh
ACTIVATION_OUTPUT = "softmax"
WEIGHT_INIT = "he"  # he | xavier

# ── Training hyperparameters ─────────────────────────────────
EPOCHS = 40
BATCH_SIZE = 128
LEARNING_RATE = 0.01
LR_DECAY = 0.97  # multiplicative decay per epoch
OPTIMIZER = "adam"  # sgd | momentum | adam
BETA1 = 0.9  # Adam
BETA2 = 0.999  # Adam
EPSILON = 1e-8
L2_LAMBDA = 1e-4  # L2 weight regularization
DROPOUT_KEEP_PROB = 0.8  # applied to hidden layers during training
EARLY_STOPPING_PATIENCE = 5
GRAD_CLIP_NORM = 5.0

# ── Hyperparameter search grid (model_tuning.py) ─────────────
TUNING_TRIALS = 6
HIDDEN_LAYER_CHOICES = [
    [128, 64],
    [256, 128, 64],
    [256, 128],
]
LR_CHOICES = [0.1, 0.05, 0.01]

# ── Challenger promotion gates ────────────────────────────────
MIN_ACCURACY_IMPROVEMENT = 0.002
MIN_MACRO_F1_THRESHOLD = 0.90
MAX_GENERALIZATION_GAP = 0.06  # train_acc - test_acc

# ── PSI drift thresholds (on mean pixel-intensity-bucket distribution) ─
PSI_MODERATE = 0.10
PSI_HIGH = 0.20

# ── Confidence / alerting ─────────────────────────────────────
LOW_CONFIDENCE_ALERT = 0.55  # avg softmax top-1 prob below this → alert

# ── Paths ───────────────────────────────────────────────────
DATA_DIR = r"D:\Data Science Datasets\ANN from Scratch — MNIST Digit Recognizer"
MODEL_DIR = "ann_models"
LOGS_DIR = "logs"
METRICS_LOG = os.path.join(MODEL_DIR, "metrics_log.csv")
TESTS_DIR = "tests"
SERVING_DIR = "serving"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
