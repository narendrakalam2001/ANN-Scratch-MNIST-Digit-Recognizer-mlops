# ============================================================
# TEST SUITE — ANN From Scratch: MNIST Digit Recognizer
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.neural_network import (
    NeuralNetworkFromScratch,
    relu,
    relu_derivative,
    leaky_relu,
    tanh,
    tanh_derivative,
    softmax,
    cross_entropy_loss,
)
from src.preprocessing import (
    PixelScaler,
    one_hot_encode,
    one_hot_decode,
    stratified_split,
    train_val_test_split,
)
from src.metrics import (
    accuracy,
    confusion_matrix_np,
    precision_recall_f1_per_class,
    macro_f1,
    top_k_accuracy,
    mean_top1_confidence,
    psi,
    generalization_gap,
)
from src.leakage_check import check_split_leakage, check_label_consistency
from src.data_loader import validate_raw_data
from src.model_loader import run_challenger_comparison, _load_champion_metrics

# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def toy_data():
    rng = np.random.RandomState(0)
    X = rng.randint(0, 256, size=(200, 784)).astype(float)
    y = rng.randint(0, 10, size=200)
    return X, y


@pytest.fixture
def tiny_network():
    return NeuralNetworkFromScratch(
        layer_dims=[784, 16, 10],
        activation="relu",
        weight_init="he",
        optimizer="adam",
        learning_rate=0.01,
        random_state=42,
    )


# ============================================================
# ACTIVATION FUNCTIONS
# ============================================================


class TestActivations:

    def test_relu_zeroes_negatives(self):
        z = np.array([[-2, -1, 0, 1, 2]])
        out = relu(z)
        assert (out >= 0).all()
        assert out.tolist() == [[0, 0, 0, 1, 2]]

    def test_relu_derivative_is_step_function(self):
        z = np.array([[-1.0, 0.0, 1.0]])
        d = relu_derivative(z)
        assert d.tolist() == [[0.0, 0.0, 1.0]]

    def test_leaky_relu_passes_small_negative_gradient(self):
        z = np.array([[-10.0]])
        out = leaky_relu(z, alpha=0.01)
        assert out[0, 0] == pytest.approx(-0.1)

    def test_tanh_bounded_between_minus_one_and_one(self):
        z = np.array([[-100.0, 0.0, 100.0]])
        out = tanh(z)
        assert out.min() >= -1.0 and out.max() <= 1.0

    def test_tanh_derivative_max_at_zero(self):
        d0 = tanh_derivative(np.array([[0.0]]))
        d5 = tanh_derivative(np.array([[5.0]]))
        assert d0[0, 0] > d5[0, 0]

    def test_softmax_rows_sum_to_one(self):
        z = np.array([[1.0, 2.0, 3.0], [0.1, 0.2, 0.7]])
        probs = softmax(z)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_softmax_numerically_stable_for_large_values(self):
        z = np.array([[1000.0, 1001.0, 1002.0]])
        probs = softmax(z)
        assert not np.isnan(probs).any()
        assert np.allclose(probs.sum(), 1.0)


# ============================================================
# LOSS
# ============================================================


class TestLoss:

    def test_cross_entropy_zero_for_perfect_prediction(self):
        y_true = np.array([[0, 1, 0]])
        y_prob = np.array([[1e-12, 1.0 - 2e-12, 1e-12]])
        loss = cross_entropy_loss(y_prob, y_true)
        assert loss < 1e-6

    def test_cross_entropy_high_for_confident_wrong_prediction(self):
        y_true = np.array([[1, 0, 0]])
        y_prob = np.array([[1e-6, 0.999999, 1e-9]])
        loss = cross_entropy_loss(y_prob, y_true)
        assert loss > 5.0

    def test_cross_entropy_is_nonnegative(self):
        rng = np.random.RandomState(1)
        y_prob = softmax(rng.randn(10, 4))
        y_true = one_hot_encode(rng.randint(0, 4, 10), num_classes=4)
        assert cross_entropy_loss(y_prob, y_true) >= 0


# ============================================================
# NEURAL NETWORK — INITIALIZATION
# ============================================================


class TestNetworkInit:

    def test_weight_shapes_match_layer_dims(self, tiny_network):
        assert tiny_network.params["W1"].shape == (784, 16)
        assert tiny_network.params["W2"].shape == (16, 10)

    def test_bias_shapes(self, tiny_network):
        assert tiny_network.params["b1"].shape == (1, 16)
        assert tiny_network.params["b2"].shape == (1, 10)

    def test_he_init_scale_reasonable(self):
        net = NeuralNetworkFromScratch(layer_dims=[784, 100, 10], weight_init="he", random_state=0)
        std = net.params["W1"].std()
        expected_std = np.sqrt(2.0 / 784)
        assert std == pytest.approx(expected_std, rel=0.3)

    def test_same_seed_gives_reproducible_weights(self):
        net1 = NeuralNetworkFromScratch(layer_dims=[784, 16, 10], random_state=42)
        net2 = NeuralNetworkFromScratch(layer_dims=[784, 16, 10], random_state=42)
        assert np.allclose(net1.params["W1"], net2.params["W1"])


# ============================================================
# FORWARD / BACKWARD PROPAGATION
# ============================================================


class TestForwardBackward:

    def test_forward_output_shape(self, tiny_network, toy_data):
        X, _ = toy_data
        probs, _ = tiny_network.forward(X, training=False)
        assert probs.shape == (200, 10)

    def test_forward_output_is_valid_probability_distribution(self, tiny_network, toy_data):
        X, _ = toy_data
        probs, _ = tiny_network.forward(X, training=False)
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert (probs >= 0).all()

    def test_backward_gradient_shapes_match_params(self, tiny_network, toy_data):
        X, y = toy_data
        y_oh = one_hot_encode(y)
        probs, cache = tiny_network.forward(X, training=True)
        grads = tiny_network.backward(cache, y_oh)
        for l in range(1, tiny_network.num_layers + 1):
            assert grads[f"dW{l}"].shape == tiny_network.params[f"W{l}"].shape
            assert grads[f"db{l}"].shape == tiny_network.params[f"b{l}"].shape

    def test_dropout_disabled_at_inference(self, tiny_network, toy_data):
        X, _ = toy_data
        probs1, _ = tiny_network.forward(X, training=False)
        probs2, _ = tiny_network.forward(X, training=False)
        assert np.allclose(probs1, probs2)  # no randomness without training=True

    def test_gradient_clipping_bounds_norm(self, tiny_network):
        grads = {
            "dW1": np.ones((784, 16)) * 100,
            "db1": np.ones((1, 16)) * 100,
            "dW2": np.ones((16, 10)) * 100,
            "db2": np.ones((1, 10)) * 100,
        }
        clipped = tiny_network._clip_gradients(grads)
        total_norm = np.sqrt(sum(np.sum(g**2) for g in clipped.values()))
        assert total_norm <= tiny_network.grad_clip_norm + 1e-3


# ============================================================
# TRAINING — loss should decrease
# ============================================================


class TestTraining:

    def test_loss_decreases_over_epochs_on_learnable_toy_problem(self):
        rng = np.random.RandomState(0)
        # Make an easily separable toy classification problem
        X = rng.randn(300, 20)
        true_w = rng.randn(20, 3)
        y = np.argmax(X @ true_w, axis=1)
        y_oh = one_hot_encode(y, num_classes=3)

        net = NeuralNetworkFromScratch(
            layer_dims=[20, 32, 3], learning_rate=0.05, dropout_keep_prob=1.0, random_state=0
        )
        history = net.fit(X, y_oh, epochs=15, batch_size=32, early_stopping_patience=15, verbose=False)

        assert history["train_loss"][-1] < history["train_loss"][0]

    def test_train_accuracy_improves_on_separable_toy_problem(self):
        rng = np.random.RandomState(0)
        X = rng.randn(300, 20)
        true_w = rng.randn(20, 3)
        y = np.argmax(X @ true_w, axis=1)
        y_oh = one_hot_encode(y, num_classes=3)

        net = NeuralNetworkFromScratch(
            layer_dims=[20, 32, 3], learning_rate=0.05, dropout_keep_prob=1.0, random_state=0
        )
        history = net.fit(X, y_oh, epochs=20, batch_size=32, early_stopping_patience=20, verbose=False)

        assert history["train_acc"][-1] > history["train_acc"][0]

    def test_predict_returns_valid_class_indices(self, tiny_network, toy_data):
        X, _ = toy_data
        preds = tiny_network.predict(X)
        assert preds.min() >= 0 and preds.max() <= 9
        assert preds.shape == (200,)


# ============================================================
# SAVE / LOAD
# ============================================================


class TestSaveLoad:

    def test_save_and_load_preserve_predictions(self, tmp_path, tiny_network, toy_data):
        X, _ = toy_data
        prefix = str(tmp_path / "test_net")
        preds_before = tiny_network.predict(X)
        tiny_network.save(prefix)

        loaded = NeuralNetworkFromScratch.load(prefix)
        preds_after = loaded.predict(X)

        assert np.array_equal(preds_before, preds_after)

    def test_save_creates_expected_files(self, tmp_path, tiny_network):
        prefix = str(tmp_path / "test_net")
        tiny_network.save(prefix)
        assert os.path.exists(f"{prefix}.npz")
        assert os.path.exists(f"{prefix}_config.json")


# ============================================================
# PREPROCESSING
# ============================================================


class TestPreprocessing:

    def test_pixel_scaler_scales_to_zero_one_range(self):
        X = np.array([[0.0, 128.0, 255.0]])
        scaler = PixelScaler()
        X_scaled = scaler.fit_transform(X)
        assert X_scaled.min() >= 0.0 and X_scaled.max() <= 1.0
        assert X_scaled[0, 2] == pytest.approx(1.0)

    def test_pixel_scaler_raises_if_not_fit(self):
        scaler = PixelScaler()
        with pytest.raises(RuntimeError):
            scaler.transform(np.array([[1.0]]))

    def test_one_hot_encode_shape_and_values(self):
        y = np.array([0, 3, 9])
        oh = one_hot_encode(y, num_classes=10)
        assert oh.shape == (3, 10)
        assert oh[0, 0] == 1.0 and oh[1, 3] == 1.0 and oh[2, 9] == 1.0
        assert oh.sum() == 3

    def test_one_hot_decode_is_inverse_of_encode(self):
        y = np.array([1, 5, 8, 0])
        assert np.array_equal(one_hot_decode(one_hot_encode(y)), y)

    def test_stratified_split_preserves_class_proportions(self, toy_data):
        X, y = toy_data
        X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2, random_state=0)
        assert len(X_train) + len(X_test) == len(X)
        # every class present in original should roughly appear in both splits
        for cls in np.unique(y):
            assert (y_train == cls).sum() > 0

    def test_train_val_test_split_sizes_correct(self, toy_data):
        X, y = toy_data
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
            X, y, val_size=0.1, test_size=0.1, random_state=0
        )
        assert len(X_train) + len(X_val) + len(X_test) == len(X)
        assert len(X_test) == pytest.approx(len(X) * 0.1, abs=2)


# ============================================================
# METRICS
# ============================================================


class TestMetrics:

    def test_accuracy_perfect_predictions(self):
        y = np.array([0, 1, 2, 3])
        assert accuracy(y, y) == 1.0

    def test_accuracy_all_wrong(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        assert accuracy(y_true, y_pred) == 0.0

    def test_confusion_matrix_diagonal_for_perfect_predictions(self):
        y = np.array([0, 1, 2])
        cm = confusion_matrix_np(y, y, num_classes=3)
        assert np.array_equal(cm, np.eye(3, dtype=int))

    def test_macro_f1_perfect_predictions_equals_one(self):
        y = np.array([0, 1, 2, 3, 4] * 5)
        assert macro_f1(y, y, num_classes=5) == pytest.approx(1.0)

    def test_precision_recall_f1_per_class_lengths(self):
        y_true = np.array([0, 1, 1, 2])
        y_pred = np.array([0, 1, 2, 2])
        p, r, f1 = precision_recall_f1_per_class(y_true, y_pred, num_classes=3)
        assert len(p) == len(r) == len(f1) == 3

    def test_top_k_accuracy_at_least_top1(self):
        y_true = np.array([0, 1])
        y_prob = np.array([[0.9, 0.1], [0.2, 0.8]])
        top1 = top_k_accuracy(y_true, y_prob, k=1)
        top2 = top_k_accuracy(y_true, y_prob, k=2)
        assert top1 == 1.0
        assert top2 >= top1

    def test_mean_top1_confidence_in_valid_range(self):
        y_prob = np.array([[0.7, 0.3], [0.6, 0.4]])
        conf = mean_top1_confidence(y_prob)
        assert 0.0 <= conf <= 1.0

    def test_psi_zero_for_identical_distributions(self):
        rng = np.random.RandomState(0)
        dist = rng.rand(500)
        assert psi(dist, dist) == pytest.approx(0.0, abs=1e-6)

    def test_psi_high_for_very_different_distributions(self):
        rng = np.random.RandomState(0)
        expected = rng.normal(0, 1, 500)
        actual = rng.normal(10, 1, 500)
        assert psi(expected, actual) > 0.2

    def test_generalization_gap_positive_when_overfitting(self):
        assert generalization_gap(0.99, 0.80) == pytest.approx(0.19)


# ============================================================
# LEAKAGE CHECK
# ============================================================


class TestLeakageCheck:

    def test_no_leakage_passes_on_disjoint_data(self):
        rng = np.random.RandomState(0)
        X_train = rng.randint(0, 256, (50, 784))
        X_val = rng.randint(0, 256, (20, 784))
        X_test = rng.randint(0, 256, (20, 784))
        report = check_split_leakage(X_train, X_val, X_test)
        assert report["train_test_overlap"] == 0

    def test_leakage_raises_on_duplicate_image(self):
        rng = np.random.RandomState(0)
        X_train = rng.randint(0, 256, (10, 784))
        X_test = X_train[:1].copy()  # duplicate injected into test
        with pytest.raises(ValueError):
            check_split_leakage(X_train, np.zeros((0, 784)), X_test)

    def test_label_consistency_flags_missing_classes(self):
        y_train = np.array([0, 1, 2])
        y_val = np.array([0, 1])
        y_test = np.array([0])
        report = check_label_consistency(y_train, y_val, y_test, num_classes=5)
        assert 3 in report["train"]["classes_missing"]


# ============================================================
# DATA VALIDATION
# ============================================================


class TestDataValidation:

    def test_validate_raw_data_passes_valid_frame(self):
        cols = ["label"] + [f"pixel{i}" for i in range(784)]
        data = np.zeros((5, 785))
        data[:, 0] = [0, 1, 2, 3, 4]
        df = pd.DataFrame(data, columns=cols)
        validated = validate_raw_data(df, has_label=True)
        assert len(validated) == 5

    def test_validate_raw_data_raises_on_missing_pixel_columns(self):
        df = pd.DataFrame({"label": [0], "pixel0": [1]})
        with pytest.raises(ValueError):
            validate_raw_data(df, has_label=True)

    def test_validate_raw_data_raises_on_bad_label(self):
        cols = ["label"] + [f"pixel{i}" for i in range(784)]
        data = np.zeros((1, 785))
        data[0, 0] = 15  # invalid digit
        df = pd.DataFrame(data, columns=cols)
        with pytest.raises(ValueError):
            validate_raw_data(df, has_label=True)


# ============================================================
# CHAMPION-CHALLENGER SYSTEM
# ============================================================


class TestChallengerSystem:

    def test_first_model_auto_promoted(self, tmp_path, monkeypatch):
        import src.model_loader as ml

        monkeypatch.setattr(ml, "MODEL_DIR", str(tmp_path))
        monkeypatch.setattr(ml, "CHALLENGER_LOG", str(tmp_path / "challenger_log.json"))

        result = ml.run_challenger_comparison(
            challenger_name="ANN_v1",
            challenger_accuracy=0.95,
            challenger_macro_f1=0.94,
            challenger_gap=0.02,
            challenger_prefix=str(tmp_path / "ANN_v1"),
        )
        assert result["decision"] == "PROMOTED"

    def test_weaker_challenger_rejected(self, tmp_path, monkeypatch):
        import src.model_loader as ml

        monkeypatch.setattr(ml, "MODEL_DIR", str(tmp_path))
        monkeypatch.setattr(ml, "CHALLENGER_LOG", str(tmp_path / "challenger_log.json"))

        ml.run_challenger_comparison(
            challenger_name="ANN_v1",
            challenger_accuracy=0.97,
            challenger_macro_f1=0.96,
            challenger_gap=0.02,
            challenger_prefix=str(tmp_path / "ANN_v1"),
            challenger_card_path=str(tmp_path / "card_v1.json"),
        )
        import json

        os.makedirs(tmp_path, exist_ok=True)
        with open(tmp_path / "card_v1.json", "w") as f:
            json.dump(
                {
                    "model_details": {"name": "ANN_v1"},
                    "metrics": {"test_accuracy": 0.97, "test_macro_f1": 0.96},
                },
                f,
            )

        result2 = ml.run_challenger_comparison(
            challenger_name="ANN_v2",
            challenger_accuracy=0.50,
            challenger_macro_f1=0.40,
            challenger_gap=0.30,
            challenger_prefix=str(tmp_path / "ANN_v2"),
        )
        assert result2["decision"] == "REJECTED"
        assert result2["gates"]["accuracy_improvement_passed"] is False
