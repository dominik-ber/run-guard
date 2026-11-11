"""Compare models on the same test athletes with the same metrics.

Run from the repo root:

    python -m RunGuard.ml_logic.compare_models

Every model is trained with 5 random seeds; the table shows mean ± std over the seeds,
because with 14 test athletes one run can differ a lot from the next.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from tensorflow import keras

from RunGuard.ml_logic.dl_model import (compile_dl_model, initialize_mlp_model,
                                        predict_dl_model, split_train_val, train_dl_model)
from RunGuard.ml_logic.model import (evaluate_injury_risk, initialize_xgboost_model, predict,
                                     train_model)
from RunGuard.ml_logic.train_test_data import train_test_data_prep

DATA_PATH = "raw_data/day_approach_maskedID_timeseries.csv"
SEEDS = [0, 1, 2, 3, 4]
METRIC_NAMES = ["pr_auc", "roc_auc", "recall_at_far_10%", "log_loss", "mean_prob"]


def run_xgboost(X_train, y_train, X_test, seed, class_weight=True):
    """Takes Ben's XGBoost model and trains with random state = [0,1,2,3,4]

    class_weight=True: keeps scale_pos_weight = neg/pos (~74.5) as in model.py;
    class_weight=False: sets it to 1, so every row counts the same, like in the MLP.
    """
    model = initialize_xgboost_model(y_train)
    model.set_params(random_state=seed)
    if not class_weight:
        model.set_params(scale_pos_weight=1)

    history, model = train_model(model, X_train, y_train)
    scores = predict(model, X_test)
    return scores


def run_mlp(X_train, y_train, groups, X_test, seed):
    """Train the MLP from dl_model.py with a given seed; return test scores."""
    # Fixes the random start of the weights, the dropout and the order of the batches
    keras.utils.set_random_seed(seed)

    # Step 1: a different seed also puts other athletes into validation
    X_tr, y_tr, X_val, y_val = split_train_val(X_train, y_train, groups, seed=seed)
    # Step 2: show_summary=False, otherwise the table of layers is printed for every seed
    model = initialize_mlp_model(X_tr, show_summary=False)
    # Step 3
    model = compile_dl_model(model)
    # Step 4
    model, history = train_dl_model(model, X_tr, y_tr, X_val, y_val)
    # Step 5
    scores = predict_dl_model(model, X_test)
    return scores


def evaluate_scores(model_name, seed, y_test, scores):
    """All metrics of one model and one seed, as one result row (a dictionary)."""
    metrics = evaluate_injury_risk(y_test, scores)

    row = {
        "model": model_name,
        "seed": seed,
        "pr_auc": metrics["pr_auc"],
        "roc_auc": metrics["roc_auc"],
        "recall_at_far_10%": metrics["recall_at_far_10%"],
        # log-loss: how good the probabilities themselves are (lower is better)
        "log_loss": log_loss(y_test, scores),
        # mean predicted probability; compare with the real injury rate
        "mean_prob": scores.mean(),
    }

    print(f"{model_name:20s} seed {seed}:  pr_auc {row['pr_auc']:.4f}  roc_auc {row['roc_auc']:.4f}")
    return row


def summarize(results):
    """One line per model: 'mean ± std' of every metric over the seeds."""
    results_df = pd.DataFrame(results)

    table_rows = []
    # unique() gives every model name once, in the order they first appear
    for model_name in results_df["model"].unique():
        # Only the rows (one per seed) of this model
        model_rows = results_df[results_df["model"] == model_name]

        table_row = {"model": model_name}
        for metric in METRIC_NAMES:
            mean = model_rows[metric].mean()
            std = model_rows[metric].std()
            table_row[metric] = f"{mean:.4f} ± {std:.4f}"
        table_rows.append(table_row)

    return pd.DataFrame(table_rows)


def main():
    data = pd.read_csv(DATA_PATH)
    X_train, y_train, X_test, y_test = train_test_data_prep(data)
    # train_test_data_prep keeps the original row numbers (the index),
    # so we can look up the athlete of every training row in the original data
    groups = data.loc[X_train.index, "Athlete ID"]

    results = []
    for seed in SEEDS:
        print(f"\n--- seed {seed} ---")

        # 0.5 for everyone: the lower bound, ROC-AUC = 0.5 (the same for every seed)
        no_skill_scores = np.full(len(X_test), 0.5)
        results.append(evaluate_scores("no-skill", seed, y_test, no_skill_scores))

        xgb_scores = run_xgboost(X_train, y_train, X_test, seed, class_weight=True)
        results.append(evaluate_scores("xgboost", seed, y_test, xgb_scores))

        xgb_no_weight_scores = run_xgboost(X_train, y_train, X_test, seed, class_weight=False)
        results.append(evaluate_scores("xgboost (no weight)", seed, y_test, xgb_no_weight_scores))

        mlp_scores = run_mlp(X_train, y_train, groups, X_test, seed)
        results.append(evaluate_scores("mlp", seed, y_test, mlp_scores))

    print(f"\nTest: {len(X_test)} rows, injury rate {y_test.mean():.4f}")
    print(summarize(results).to_string(index=False))


if __name__ == "__main__":
    main()
