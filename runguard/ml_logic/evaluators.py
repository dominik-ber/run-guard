import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

def evaluate_injury_risk_model(y_true, y_scores, target_fpr=0.10):
    """
    Evaluates risk scores against true binary labels.

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 for non-injury, 1 for injury).
    y_scores : array-like of shape (n_samples,)
        Predicted risk scores or probabilities from the model.
    target_fpr : float, default=0.10
        The target false-alarm rate (False Positive Rate) to evaluate recall at.

    Returns:
    --------
    metrics : dict
        A dictionary containing ROC-AUC, PR-AUC, the recall at the target FPR,
        and the actual achieved FPR at that threshold.
    """
    # 1. Compute ROC-AUC
    roc_auc = roc_auc_score(y_true, y_scores)

    # 2. Compute PR-AUC (Average Precision is a robust approximation of the area under the PR curve)
    pr_auc = average_precision_score(y_true, y_scores)

    # 3. Compute Recall at fixed False-Alarm Rate (FPR)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Find the threshold index where FPR is closest to the target_fpr
    idx = np.argmin(np.abs(fpr - target_fpr))

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "recall_at_target_fpr": float(tpr[idx]),
        "actual_fpr": float(fpr[idx]), # Helpful to verify closeness
        "threshold": float(thresholds[idx])
    }
