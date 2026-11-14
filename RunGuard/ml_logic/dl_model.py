"""Deep learning models (Keras) for injury prediction, to compare with XGBoost in model.py.

Same input as XGBoost: the 28 contract columns from `train_test_data_prep`.
Same evaluation: `evaluate_injury_risk` on the same test athletes.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from RunGuard.ml_logic.mlp_numpy import MLP_WEIGHTS_PATH


def split_train_val(X_train: pd.DataFrame, y_train: pd.Series, groups: pd.Series,
                    val_share: float = 0.15, seed: int = 42):
    """Split the training data into train and validation parts by athlete.

    X_train, y_train — that is returned by Dominik's train_test_data_prep

    groups - athlete's Ids for every row, train_test_data_prep doesn't return
    them, so we fetch them by data.loc[X_train.index, "Athlete ID"]

    val_share=0.15 - proportion of athletes in val set

    seed=42 - guarantees that split will be reproduced

    The validation part is used for early stopping, so the test set stays untouched.
    Whole athletes go to one side: consecutive rows of one athlete share 6 of 7 days,
    so a random row split would leak them into validation.

    Returns:
    X_tr, y_tr, X_val, y_val
    """
    # if we have 0.15 rate, we will get 7 folds
    n_splits = round(1 / val_share)
    # same tool which was used in train_test_data_prep
    # "group" means that a whole athlete will go to the single fold
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # next takes the first fold as val, others as train
    tr_idx, val_idx = next(sgkf.split(X_train, y_train, groups))
    # this is a check, that gurantees that a single athlete didn't get to both sets
    assert set(groups.iloc[tr_idx]).isdisjoint(groups.iloc[val_idx]), "Athlete leakage!"

    return (X_train.iloc[tr_idx], y_train.iloc[tr_idx],
            X_train.iloc[val_idx], y_train.iloc[val_idx])


def initialize_mlp_model(X_train: pd.DataFrame, l2: float = 1e-4, dropout: float = 0.3,
                         show_summary: bool = True):
    """Build a small MLP: 28 inputs -> 32 dense layers-> 16 dense layers-> 1 (injury probability).

    X_train: training features; used for the input size and to fit the scaling.

    l2=1e-4: L2 penalty on the weights (defult value is enough to prevent overfitting).

    dropout=0.3: share of neurons switched off at random in each training step.

    XJBoost (our main model) can use non-normalized data. DL model can't.
    So I've added a normalization as a first layes

    show_summary=True: print the table of layers; set False when building many models.

    Returns:
    An uncompiled Keras model.
    """
    normalizer = layers.Normalization()
    normalizer.adapt(X_train.to_numpy(dtype="float32"))
    # dropout and l2 protects the model from overfitting
    model = keras.Sequential([
        keras.Input(shape=(X_train.shape[1],)),
        normalizer,
        layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2)),
        layers.Dropout(dropout),
        layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(l2)),
        layers.Dropout(dropout),
        # sigmoid squeezes the output into 0..1, read as the injury probability
        layers.Dense(1, activation="sigmoid"),
    ])
    if show_summary:
        model.summary()
    return model


def compile_dl_model(model, learning_rate: float = 1e-3):
    """Set the loss, the optimizer and the metrics shown during training.

    model: returned by initialize_mlp_model

    learning_rate=1e-3: step size of the Adam optimizer

    binary_crossentropy is the log-loss, the same loss XGBoost uses; without class
    weights it makes the model output realistic probabilities.
    PR-AUC will be used for early stopping functionality.

    Returns:
    The compiled model.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc"),
                 keras.metrics.AUC(curve="ROC", name="roc_auc")],
    )
    return model


def train_dl_model(model, X_tr, y_tr, X_val, y_val, batch_size: int = 256, epochs: int = 200,
                   patience: int = 15, pos_weight=None, verbose: int = 0):
    """Train with early stopping on the validation PR-AUC.

    Training stops when the validation PR-AUC has not improved for `patience` epochs,
    and the weights of the best epoch are put back, so `epochs` is only an upper limit.

    model: compiled model from `compile_dl_model`.

    X_tr, y_tr, X_val, y_val: from `split_train_val`.

    batch_size=256: rows per training step; 256 gives ~3-4 injuries per batch.

    epochs=200: upper limit of passes over the training data.

    patience=15: epochs without improvement before training stops.

    pos_weight=None:  no class weights (realistic probabilities). A number such as 10
        makes every injury count 10 times in the loss; probabilities then come out too high.
    verbose=0: 0 = silent, 1 = progress bar, 2 = one line per epoch.

    Returns:
    model, history (history.history holds the loss and metrics of every epoch)
    """
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_pr_auc", mode="max", patience=patience, restore_best_weights=True)

    class_weight = None if pos_weight is None else {0: 1.0, 1: float(pos_weight)}

    history = model.fit(
        X_tr.to_numpy(dtype="float32"), y_tr.to_numpy(dtype="float32"),
        validation_data=(X_val.to_numpy(dtype="float32"), y_val.to_numpy(dtype="float32")),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping],
        class_weight=class_weight,
        verbose=verbose,
    )
    return model, history


def predict_dl_model(model, X: pd.DataFrame) -> np.ndarray:
    """Predicted injury probabilities, one per row (same output as `predict` in model.py).

    model: trained model from `train_dl_model`.
    X: the 28 contract columns, raw (the model scales them itself).

    Returns:
    np.array of shape (n,): injury probability for every row.
    """
    # Keras returns shape (n, 1); ravel() turns it into a flat array of n values
    return model.predict(X.to_numpy(dtype="float32"), verbose=0).ravel()


def export_mlp_weights(model, feature_names, path=MLP_WEIGHTS_PATH):
    """Execution of an additional model from our UI will need tensorflow
    import (this one is large) and will take significant time.

    To avoid this problem Claude offer me an option with an import of model weights
    instead of a whole model import.

    Together we have 1473 weight params and 57 normalization params.
    So as file it should take near 10 Kb

    feature_names: the column names the model was trained on, in the same order
        (X_train.columns). numpy only sees the column positions, so we save the
        names too and check them before every prediction.

    path: where to save the file; by default the file the API reads
        (MLP_WEIGHTS_PATH from mlp_numpy.py). The folder is created if it does not exist.

    Returns:
    The path of the saved file.
    """
    # model.layers: Normalization, Dense, Dropout, Dense, Dropout, Dense.
    # Dropout has no weights and is switched off for predictions, so we skip it.
    normalizer = model.layers[0]
    dense_1 = model.layers[1]
    dense_2 = model.layers[3]
    dense_3 = model.layers[5]

    # Check that the model has the layers we expect (built by initialize_mlp_model)
    assert isinstance(normalizer, layers.Normalization), "Layer 0 must be Normalization"
    assert isinstance(dense_1, layers.Dense), "Layer 1 must be Dense"
    assert isinstance(dense_2, layers.Dense), "Layer 3 must be Dense"
    assert isinstance(dense_3, layers.Dense), "Layer 5 must be Dense"

    # Normalization keeps 3 values: mean, variance and the row count (count is not needed)
    normalizer_weights = normalizer.get_weights()
    mean = normalizer_weights[0]
    variance = normalizer_weights[1]

    # Every Dense layer keeps 2 arrays: the weight matrix W and the bias vector b
    W1, b1 = dense_1.get_weights()
    W2, b2 = dense_2.get_weights()
    W3, b3 = dense_3.get_weights()

    # dtype=str stores the names as plain text; without it pandas column names become
    # an "object" array, which np.load refuses to read (unless allow_pickle=True)
    feature_names = np.array(list(feature_names), dtype=str)

    # Create the folder, then save all arrays into one file, each under its own name
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path,
             feature_names=feature_names,
             mean=mean, variance=variance,
             W1=W1, b1=b1,
             W2=W2, b2=b2,
             W3=W3, b3=b3)

    print(f"MLP weights saved in: {path}")
    return path
