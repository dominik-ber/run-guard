import xgboost as xgb
import pandas as pd
import pickle
from pathlib import Path
import time
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve,auc,precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt

from RunGuard.ml_logic.train_test_data import train_test_data_prep



def train_model(model, X_train, y_train):
    history = model.fit(X_train, y_train)
    return history, model

def load_model(model_name=None):
    """
    Load a trained model from a pickle file in the 'models' directory.
    Parameters:
    model_name (str, optional): Name of the model file to be loaded (e.g., 'model_20230101-123456.pkl').
        If None, the most recently saved model is loaded.
    Returns:
    Trained XGBoost model.
    """

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    models_dir = PROJECT_ROOT / "models"

    if model_name is None:
        model_paths = sorted(models_dir.glob("model_*.pkl"))
        if not model_paths:
            raise FileNotFoundError(f"No model found in '{models_dir}'. Run: python -m RunGuard.ml_logic.model")
        model_path = model_paths[-1]
    else:
        model_path = models_dir / model_name

    if not model_path.exists():
        raise FileNotFoundError(f"Model file '{model_name}' not found in '{models_dir}'.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model

def save_model(model):
    """
    Save the trained model to a file with a timestamp in the 'models' directory.
    The model is saved in pickle format.
    Parameters:
    model: Trained XGBoost model to be saved.
    Returns:
    None
    """

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    with open(models_dir / f"model_{timestamp}.pkl", "wb") as f:
        pickle.dump(model, f)

    return print(f"Model saved in : {models_dir / f'model_{timestamp}.pkl'}")

def evaluate_model() :
    """
    TODO Evaluate trained model performance on the dataset
    """

    return None

def initialize_xgboost_model(y_train) :
    """
    Train an XGBoost model on the provided training data

    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target.

    Returns:
    xgboost model
    """


    # 2. Compute imbalance ratio for class weighting
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight_val = neg_count / pos_count

    # es = EarlyStopping() / TODO


    # Paramètres de base XGBoost
    baseline_params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",  # Évalue directement la PR-AUC (idéal pour l'imbalance)
        "tree_method": "hist",  # Algorithme rapide et moderne basé sur des histogrammes
        "scale_pos_weight": scale_pos_weight_val,  # LE paramètre clé pour rééquilibrer les classes
        "learning_rate": 0.1,  # Vitesse d'apprentissage standard (compromis stabilité/vitesse)
        "max_depth": 5,  # Profondeur modérée pour éviter le surapprentissage (généralement entre 4 et 6)
        "subsample": 0.8,  # Utilise 80% des lignes par arbre (ajoute de la diversité)
        "colsample_bytree": 0.8,  # Utilise 80% des colonnes/features par arbre
        "n_estimators": 200,  # Nombre d'arbres de départ (à coupler idéalement avec l'early stopping)
        "random_state": 42,  # Fixe la graine pour la reproductibilité
        "n_jobs": -1,  # Utilise tous les cœurs du processeur
    }



    # 3. Initialize XGBoost with base_line params
    model = xgb.XGBClassifier(**baseline_params)

    # 5. Score / predict probabilities on the test table
    # y_scores = model.predict_proba(X_test)[:, 1]

    # 6. Evaluate using the previously defined plot_roc_pr_curves function
    #plot_roc_pr_curves(y_test, y_scores)

    return model

### put here everything for the model: initialize, compile, train, evaliate, predict
### save function for trained model in the cloud (ml flow?)
### load function to load the model later in the api
def predict(model, X_test):
    """
    Predict using the trained model

    Parameters:
    model: Trained XGBoost model.
    X_test (pd.DataFrame): Test features.

    Returns:
    np.array: Predicted probabilities for the positive class.
    """
    y_scores = model.predict_proba(X_test)[:, 1]
    return y_scores


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


if __name__ == "__main__":

    data = pd.read_csv("raw_data/day_approach_maskedID_timeseries.csv")
    X_train, y_train, X_test, y_test = train_test_data_prep(data)

    model = initialize_xgboost_model(y_train)
    history, model = train_model(model, X_train, y_train)

    print(evaluate_injury_risk(y_test, predict(model, X_test)))
    save_model(model)
