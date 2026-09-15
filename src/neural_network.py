# ============================================================
# NEURAL NETWORK FROM SCRATCH — NumPy only, no frameworks
# ANN From Scratch: MNIST Digit Recognizer
# ============================================================
#
# Implements, by hand, everything a framework normally hides:
#   - He / Xavier weight initialization
#   - Forward propagation (Linear -> Activation -> ... -> Softmax)
#   - Cross-entropy loss (+ L2 regularization)
#   - Backward propagation (manual chain-rule gradients)
#   - Mini-batch gradient descent with SGD / Momentum / Adam
#   - Inverted dropout regularization
#   - Gradient clipping
#   - Learning-rate decay + early stopping
#
# Every matrix operation below is explicit so the math is
# auditable line-by-line — this is the whole point of the
# project (interview-proof understanding of backprop).
# ============================================================

import numpy as np
import logging
import time

logger = logging.getLogger(__name__)


# ============================================================
# ACTIVATION FUNCTIONS + DERIVATIVES
# ============================================================


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(z.dtype)


def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)


def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha).astype(z.dtype)


def tanh(z):
    return np.tanh(z)


def tanh_derivative(z):
    return 1.0 - np.tanh(z) ** 2


def softmax(z):
    """Numerically stable softmax — subtract row-max before exponentiating."""
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


ACTIVATIONS = {
    "relu": (relu, relu_derivative),
    "leaky_relu": (leaky_relu, leaky_relu_derivative),
    "tanh": (tanh, tanh_derivative),
}


# ============================================================
# LOSS
# ============================================================


def cross_entropy_loss(y_prob: np.ndarray, y_true_onehot: np.ndarray, eps: float = 1e-12) -> float:
    """
    Categorical cross-entropy over a batch.
    y_prob        : (batch, num_classes) softmax outputs
    y_true_onehot : (batch, num_classes) one-hot targets
    """
    m = y_true_onehot.shape[0]
    y_prob_clipped = np.clip(y_prob, eps, 1.0 - eps)
    loss = -np.sum(y_true_onehot * np.log(y_prob_clipped)) / m
    return float(loss)


# ============================================================
# NEURAL NETWORK — fully connected, from scratch
# ============================================================


class NeuralNetworkFromScratch:
    """
    Fully-connected feed-forward ANN implemented with raw NumPy.

    Architecture: input -> [Linear -> Activation -> Dropout]*N -> Linear -> Softmax

    Parameters
    ----------
    layer_dims : list[int]
        e.g. [784, 256, 128, 64, 10] -> input, hidden..., output
    activation : str
        'relu' | 'leaky_relu' | 'tanh'  (applied to all hidden layers)
    weight_init : str
        'he' (good for relu) | 'xavier' (good for tanh)
    optimizer : str
        'sgd' | 'momentum' | 'adam'
    """

    def __init__(
        self,
        layer_dims,
        activation="relu",
        weight_init="he",
        optimizer="adam",
        learning_rate=0.01,
        l2_lambda=1e-4,
        dropout_keep_prob=0.8,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        grad_clip_norm=5.0,
        random_state=42,
    ):
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims) - 1  # count of weight matrices
        self.activation_name = activation
        self.act_fn, self.act_deriv_fn = ACTIVATIONS[activation]
        self.weight_init = weight_init
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.dropout_keep_prob = dropout_keep_prob
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.grad_clip_norm = grad_clip_norm
        self.random_state = random_state

        self._rng = np.random.RandomState(random_state)
        self.params = self._initialize_parameters()

        # Adam / momentum moment buffers
        self._m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._t = 0  # Adam timestep

        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # ------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------
    def _initialize_parameters(self):
        """He init for ReLU family, Xavier/Glorot for tanh — prevents
        vanishing/exploding activations at the start of training."""
        params = {}
        for l in range(1, self.num_layers + 1):
            fan_in = self.layer_dims[l - 1]
            fan_out = self.layer_dims[l]

            if self.weight_init == "he":
                scale = np.sqrt(2.0 / fan_in)
            else:  # xavier
                scale = np.sqrt(1.0 / fan_in)

            params[f"W{l}"] = self._rng.randn(fan_in, fan_out) * scale
            params[f"b{l}"] = np.zeros((1, fan_out))

        return params

    # ------------------------------------------------------------
    # FORWARD PROPAGATION
    # ------------------------------------------------------------
    def forward(self, X, training=True):
        """
        Returns final softmax probabilities and caches every
        intermediate value needed for backprop.
        """
        cache = {"A0": X}
        A = X

        for l in range(1, self.num_layers + 1):
            W, b = self.params[f"W{l}"], self.params[f"b{l}"]
            Z = A @ W + b
            cache[f"Z{l}"] = Z

            if l < self.num_layers:
                A = self.act_fn(Z)

                # Inverted dropout — only during training, only on hidden layers
                if training and self.dropout_keep_prob < 1.0:
                    keep_prob = self.dropout_keep_prob
                    mask = (self._rng.rand(*A.shape) < keep_prob).astype(A.dtype) / keep_prob
                    A = A * mask
                    cache[f"D{l}"] = mask
            else:
                A = softmax(Z)  # output layer

            cache[f"A{l}"] = A
            A = cache[f"A{l}"]

        return A, cache

    # ------------------------------------------------------------
    # BACKWARD PROPAGATION — manual chain rule, layer by layer
    # ------------------------------------------------------------
    def backward(self, cache, y_onehot):
        m = y_onehot.shape[0]
        grads = {}

        # ---- Output layer: dL/dZ_out = y_hat - y  (softmax + cross-entropy combined gradient) ----
        A_out = cache[f"A{self.num_layers}"]
        dZ = (A_out - y_onehot) / m  # (batch, num_classes)

        for l in range(self.num_layers, 0, -1):
            A_prev = cache[f"A{l - 1}"]
            W = self.params[f"W{l}"]

            grads[f"dW{l}"] = A_prev.T @ dZ + (self.l2_lambda / m) * W  # + L2 regularization term
            grads[f"db{l}"] = np.sum(dZ, axis=0, keepdims=True)

            if l > 1:
                dA_prev = dZ @ W.T

                # Undo dropout mask applied at this layer during forward pass
                if f"D{l - 1}" in cache:
                    dA_prev = dA_prev * cache[f"D{l - 1}"]

                Z_prev = cache[f"Z{l - 1}"]
                dZ = dA_prev * self.act_deriv_fn(Z_prev)

        return grads

    # ------------------------------------------------------------
    # GRADIENT CLIPPING
    # ------------------------------------------------------------
    def _clip_gradients(self, grads):
        total_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
        if total_norm > self.grad_clip_norm:
            scale = self.grad_clip_norm / (total_norm + 1e-8)
            for k in grads:
                grads[k] *= scale
        return grads

    # ------------------------------------------------------------
    # PARAMETER UPDATE — SGD / Momentum / Adam, hand-rolled
    # ------------------------------------------------------------
    def _update_parameters(self, grads):
        grads = self._clip_gradients(grads)
        self._t += 1

        for l in range(1, self.num_layers + 1):
            for p in ("W", "b"):
                key = f"{p}{l}"
                grad = grads[f"d{key}"]

                if self.optimizer == "sgd":
                    self.params[key] -= self.learning_rate * grad

                elif self.optimizer == "momentum":
                    self._m[key] = self.beta1 * self._m[key] + (1 - self.beta1) * grad
                    self.params[key] -= self.learning_rate * self._m[key]

                elif self.optimizer == "adam":
                    self._m[key] = self.beta1 * self._m[key] + (1 - self.beta1) * grad
                    self._v[key] = self.beta2 * self._v[key] + (1 - self.beta2) * (grad**2)

                    m_hat = self._m[key] / (1 - self.beta1**self._t)
                    v_hat = self._v[key] / (1 - self.beta2**self._t)

                    self.params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                else:
                    raise ValueError(f"Unknown optimizer: {self.optimizer}")

    # ------------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------------
    def predict_proba(self, X):
        probs, _ = self.forward(X, training=False)
        return probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    # ------------------------------------------------------------
    # TRAIN LOOP — mini-batch gradient descent, LR decay, early stopping
    # ------------------------------------------------------------
    def fit(
        self,
        X_train,
        y_train_onehot,
        X_val=None,
        y_val_onehot=None,
        epochs=40,
        batch_size=128,
        lr_decay=0.97,
        early_stopping_patience=5,
        verbose=True,
        log_every=5,
    ):
        n = X_train.shape[0]
        best_val_loss = np.inf
        best_params = None
        patience_ctr = 0
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            # ── shuffle each epoch ──
            perm = self._rng.permutation(n)
            X_shuf, y_shuf = X_train[perm], y_train_onehot[perm]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                end = start + batch_size
                X_batch = X_shuf[start:end]
                y_batch = y_shuf[start:end]

                y_prob, cache = self.forward(X_batch, training=True)
                batch_loss = cross_entropy_loss(y_prob, y_batch) + self._l2_penalty(X_batch.shape[0])
                grads = self.backward(cache, y_batch)
                self._update_parameters(grads)

                epoch_loss += batch_loss
                n_batches += 1

            self.learning_rate *= lr_decay  # decay LR each epoch

            train_loss = epoch_loss / max(n_batches, 1)
            train_acc = self._accuracy(X_train, y_train_onehot)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            log_line = f"Epoch {epoch:3d}/{epochs}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"

            if X_val is not None:
                val_prob = self.predict_proba(X_val)
                val_loss = cross_entropy_loss(val_prob, y_val_onehot)
                val_acc = float(np.mean(np.argmax(val_prob, axis=1) == np.argmax(y_val_onehot, axis=1)))
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                log_line += f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"

                # ── early stopping on val_loss ──
                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    best_params = {k: v.copy() for k, v in self.params.items()}
                    patience_ctr = 0
                else:
                    patience_ctr += 1

            if verbose and (epoch % log_every == 0 or epoch == 1 or epoch == epochs):
                logger.info(log_line)

            if X_val is not None and patience_ctr >= early_stopping_patience:
                logger.info("Early stopping triggered at epoch %d (best val_loss=%.4f)", epoch, best_val_loss)
                break

        if best_params is not None:
            self.params = best_params  # restore best checkpoint

        elapsed = time.time() - start_time
        logger.info("Training complete in %.1fs (%d epochs run)", elapsed, epoch)
        return self.history

    def _l2_penalty(self, batch_size):
        ssum = sum(np.sum(self.params[f"W{l}"] ** 2) for l in range(1, self.num_layers + 1))
        return (self.l2_lambda / (2 * batch_size)) * ssum

    def _accuracy(self, X, y_onehot):
        preds = self.predict(X)
        true = np.argmax(y_onehot, axis=1)
        return float(np.mean(preds == true))

    # ------------------------------------------------------------
    # SERIALIZATION — pure NumPy, no joblib/pickle of framework objects
    # ------------------------------------------------------------
    def get_config(self):
        return {
            "layer_dims": self.layer_dims,
            "activation": self.activation_name,
            "weight_init": self.weight_init,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "l2_lambda": self.l2_lambda,
            "dropout_keep_prob": self.dropout_keep_prob,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "grad_clip_norm": self.grad_clip_norm,
            "random_state": self.random_state,
        }

    def save(self, path_prefix):
        """Saves params as .npz + config as .json side-car."""
        import json

        np.savez(f"{path_prefix}.npz", **self.params)
        with open(f"{path_prefix}_config.json", "w") as f:
            json.dump(self.get_config(), f, indent=2)

    @classmethod
    def load(cls, path_prefix):
        import json

        with open(f"{path_prefix}_config.json") as f:
            cfg = json.load(f)
        net = cls(
            layer_dims=cfg["layer_dims"],
            activation=cfg["activation"],
            weight_init=cfg["weight_init"],
            optimizer=cfg["optimizer"],
            learning_rate=cfg["learning_rate"],
            l2_lambda=cfg["l2_lambda"],
            dropout_keep_prob=cfg["dropout_keep_prob"],
            beta1=cfg["beta1"],
            beta2=cfg["beta2"],
            epsilon=cfg["epsilon"],
            grad_clip_norm=cfg["grad_clip_norm"],
            random_state=cfg["random_state"],
        )
        loaded = np.load(f"{path_prefix}.npz")
        net.params = {k: loaded[k] for k in loaded.files}
        return net
