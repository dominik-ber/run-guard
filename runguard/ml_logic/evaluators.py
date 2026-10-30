import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve,auc,precision_recall_curve

import matplotlib.pyplot as plt

def get_simple_rule_baseline(df):
    prev_2w_avg = (df['total km.1'] + df['total km.2']) / 2.0

    score = df['total km'] / prev_2w_avg
    y_true = df["injury"]
    return {
        'y_true':y_true,
        'scores':score}

def get_noskill_baseline(df):

    CONSTANT_SCORE = 0.5
    y_true = df["injury"]

    no_skill_scores = np.ones(len(df)) * CONSTANT_SCORE

    return {
        'y_true':y_true,
        'scores':no_skill_scores
        }

def evaluate_injury_risk(y_true, scores):
    """
    Evaluate injury risk scores.

    Parameters:
    - y_true: array-like, true binary labels (0 = healthy, 1 = injury)
    - scores: array-like, predicted risk scores (or probabilities)

    Returns:
    - dict containing the PR-AUC, ROC-AUC, and Recall at the target FAR.
    """

    target_far=0.10 # false alarm rate (FAR)
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    # 1. PR-AUC (Average Precision score)
    pr_auc = average_precision_score(y_true, scores)

    # 2. ROC-AUC
    roc_auc = roc_auc_score(y_true, scores)

    # 3. Recall at a fixe False Alarm Rate (FAR / FPR)
    fpr, tpr, _ = roc_curve(y_true, scores)
    valid_indices = np.where(fpr <= target_far)[0]
    recall_at_far = tpr[valid_indices[-1]]

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "recall_at_far_10%": recall_at_far
    }

def plot_roc_pr_curves(y_true, y_scores):
  """
  Plot the ROC-AUC and PR-AUC curves side by side.

  Parameters:
  - y_true : Trues labels (0 ou 1)
  - y_scores : Predicted probabilities for the positive class (between 0 and 1)
  """
  # 1. ROC curve calcul
  fpr, tpr, _ = roc_curve(y_true, y_scores)
  roc_auc = auc(fpr, tpr)

  # 2. Precision-Recall curve calcul
  precision, recall, _ = precision_recall_curve(y_true, y_scores)
  pr_auc = average_precision_score(y_true, y_scores)

  _, axes = plt.subplots(1, 2, figsize=(14, 6))

  # --- plot 1 : ROC-AUC ---
  axes[0].plot(
      fpr,
      tpr,
      color='darkorange',
      lw=2,
      label=f'ROC curve (AUC = {roc_auc:.3f})',
  )
  axes[0].plot(
      [0, 1],
      [0, 1],
      color='navy',
      lw=2,
      linestyle='--',
      label='Random (0.500)',
  )
  axes[0].set_xlim([0.0, 1.0])
  axes[0].set_ylim([0.0, 1.05])
  axes[0].set_xlabel('False positive rate (FPR)', fontsize=12)
  axes[0].set_ylabel('True positive rate (TPR / Recall)', fontsize=12)
  axes[0].set_title('ROC curve', fontsize=14, fontweight='bold')
  axes[0].legend(loc='lower right', fontsize=11)
  axes[0].grid(alpha=0.3)

  # --- plot 2 : PR-AUC ---
  axes[1].plot(
      recall,
      precision,
      color='blue',
      lw=2,
      label=f'PR curve (AP = {pr_auc:.3f})',
  )
  axes[1].set_xlim([0.0, 1.0])
  axes[1].set_ylim([0.0, 1.05])
  axes[1].set_xlabel('Recall', fontsize=12)
  axes[1].set_ylabel('Precision', fontsize=12)
  axes[1].set_title('Precision-Recall curve (PR)', fontsize=14, fontweight='bold')
  axes[1].legend(loc='lower left', fontsize=11)
  axes[1].grid(alpha=0.3)

  plt.tight_layout()
  plt.show()






# Exemple

if __name__ == "__main__":

    df = pd.read_csv("injury_prediction_dataset.csv")
    no_skill_scores = get_noskill_baseline(df)
    simple_rule_scores = get_simple_rule_baseline(df)

    plot_roc_pr_curves(no_skill_scores['y_true'], no_skill_scores['scores'])
