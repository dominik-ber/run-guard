"""MLP predictions with numpy only, without TensorFlow.

Our API needs only MLP model predictions, which consist by a few matrix operation.
So we can avoid importing of tensorflow (near 1 GB)

Here actually we can see a whole row of calculations
upon trained classic neural network: manually done normalization, relu, sigmoid,
weight matrix multiplication

"""

from pathlib import Path

import numpy as np
import pandas as pd

# Absolute path built from this file's location, so it works from any working folder
# (repo root, uvicorn, Docker): RunGuard/ml_logic/../mlp_weights/mlp_v1.npz
MLP_WEIGHTS_PATH = Path(__file__).parent.parent / "mlp_weights" / "mlp_v1.npz"

# The small number Keras uses to avoid dividing by zero in the Normalization layer
EPSILON = 1e-7


def load_mlp_weights(path=MLP_WEIGHTS_PATH) -> dict:
    """Read the weights file saved by `export_mlp_weights`.

    path: the .npz file; by default RunGuard/mlp_weights/mlp_v1.npz

    Returns:
    A dictionary {name: numpy array}: feature_names, mean, variance, W1, b1, W2, b2, W3, b3.
    """
    weights = {}
    # np.load opens the file; "with" closes it again after the arrays are copied out
    with np.load(path) as saved:
        for name in saved.files:
            weights[name] = saved[name]
    return weights


def relu(z: np.ndarray) -> np.ndarray:
    """ReLU activation: negative values become 0, positive values stay as they are."""
    return np.maximum(z, 0)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid activation: turns log-odds into a probability between 0 and 1."""
    return 1 / (1 + np.exp(-z))


def predict_mlp_numpy(weights: dict, X: pd.DataFrame) -> np.ndarray:
    """Predicted injury probabilities, one per row (same output as `predict_dl_model`).

    weights: returned by `load_mlp_weights`.
    X: the 28 contract columns, raw, in the same order as during training.

    Returns:
    np.array of shape (n,): injury probability for every row.
    """
    # numpy only sees positions, not names, so the columns must be in the training order
    expected_columns = list(weights["feature_names"])
    if list(X.columns) != expected_columns:
        raise ValueError(f"Columns do not match the MLP training columns: {expected_columns}")

    x = X.to_numpy(dtype="float32")

    # Layer 0, Normalization: the same scaling Keras learned with adapt() on train
    x = (x - weights["mean"]) / np.maximum(np.sqrt(weights["variance"]), EPSILON)

    # Layers 1 and 3, Dense + ReLU (Dropout layers do nothing at prediction time)
    # np.matmul = matrix multiplication: every neuron gets a weighted sum of all inputs.
    # The bias b is then added to every row.

    # Layer 1: (n, 28) x (28, 32) -> (n, 32)
    x = np.matmul(x, weights["W1"]) + weights["b1"]
    x = relu(x)

    # Layer 3: (n, 32) x (32, 16) -> (n, 16)
    x = np.matmul(x, weights["W2"]) + weights["b2"]
    x = relu(x)

    # Layer 5, Dense + sigmoid: (n, 16) x (16, 1) -> (n, 1), one probability per row
    x = np.matmul(x, weights["W3"]) + weights["b3"]
    probabilities = sigmoid(x)

    # Shape (n, 1) -> flat array of n values
    return probabilities.ravel()
