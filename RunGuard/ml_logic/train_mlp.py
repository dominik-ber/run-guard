"""
Run from the repo root (needs TensorFlow):

    python -m RunGuard.ml_logic.train_mlp

We need this file for a local execution and training of
MLP model and export of weights.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from tensorflow import keras

from RunGuard.ml_logic.dl_model import (compile_dl_model, export_mlp_weights,
                                        initialize_mlp_model, predict_dl_model,
                                        split_train_val, train_dl_model)
from RunGuard.ml_logic.mlp_numpy import load_mlp_weights, predict_mlp_numpy
from RunGuard.ml_logic.model import evaluate_injury_risk
from RunGuard.ml_logic.train_test_data import train_test_data_prep

DATA_PATH = "raw_data/day_approach_maskedID_timeseries.csv"

# Chosen in advance. We do not try several seeds and keep the best test score:
# that would tune the model on the test athletes and make the test result too optimistic.
SEED = 42


def main():
    data = pd.read_csv(DATA_PATH)
    X_train, y_train, X_test, y_test = train_test_data_prep(data)
    # Athlete of every training row, needed to split off validation athletes
    groups = data.loc[X_train.index, "Athlete ID"]

    # The same 5 steps as run_mlp in compare_models.py
    keras.utils.set_random_seed(SEED)
    X_tr, y_tr, X_val, y_val = split_train_val(X_train, y_train, groups, seed=SEED)
    model = initialize_mlp_model(X_tr)
    model = compile_dl_model(model)
    model, history = train_dl_model(model, X_tr, y_tr, X_val, y_val)

    # Early stopping put back the weights of the epoch with the best validation PR-AUC
    val_pr_auc = history.history["val_pr_auc"]
    best_epoch = int(np.argmax(val_pr_auc)) + 1  # +1 because epochs are counted from 1
    print(f"Trained {len(val_pr_auc)} epochs, best epoch {best_epoch} "
          f"(validation PR-AUC {max(val_pr_auc):.4f})")

    # Evaluate on the test athletes, the same metrics as in MODEL_COMPARISON.md
    scores = predict_dl_model(model, X_test)
    metrics = evaluate_injury_risk(y_test, scores)
    print("Test set evaluation:")
    print(f"  PR-AUC            {metrics['pr_auc']:.4f}")
    print(f"  ROC-AUC           {metrics['roc_auc']:.4f}")
    print(f"  Recall @ 10% FAR  {metrics['recall_at_far_10%']:.4f}")
    print(f"  Log-loss          {log_loss(y_test, scores):.4f}")
    print(f"  Mean probability  {scores.mean():.2%} (real injury rate {y_test.mean():.2%})")

    # Save the weights for the API
    path = export_mlp_weights(model, X_train.columns)

    # Check: the saved file with numpy must give the same predictions as Keras
    weights = load_mlp_weights(path)
    numpy_scores = predict_mlp_numpy(weights, X_test)
    max_difference = np.abs(numpy_scores - scores).max()
    print(f"Max difference numpy vs Keras: {max_difference:.1e}")
    assert max_difference < 1e-5, "numpy predictions differ from Keras, check mlp_numpy.py"


if __name__ == "__main__":
    main()
