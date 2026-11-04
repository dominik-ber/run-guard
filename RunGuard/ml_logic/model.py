import xgboost as xgb
import pandas as pd
import pickle
from pathlib import Path
import time


def train_model(model, X_train, y_train):
    history = model.fit(X_train, y_train)
    return history, model

def load_model(model_name):
    """
    Load a trained model from a pickle file in the 'models' directory.
    Parameters:
    model_name (str): Name of the model file to be loaded (e.g., 'model_20230101-123456.pkl').
    Returns:
    Trained XGBoost model.
    """

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    models_dir = PROJECT_ROOT / "models"
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
