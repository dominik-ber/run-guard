import numpy as np
import pandas as pd

def generate_injury_dataset(n_rows=1000, seed=42):
    """
    Génère un jeu de données synthétique pour la prédiction de blessures
    avec la structure de colonnes en format large (t, t-1, t-2, ..., t-6).
    """
    np.random.seed(seed)

    data = {}

    # 1. Génération des variables pour la semaine actuelle (t) et les 6 semaines précédentes (.1 à .6)
    # On simule des entraînements de course à pied, renforcement musculaire, etc.
    for i in range(7):
        suffix = f".{i}" if i > 0 else ""

        # Nombre de séances (ex: 2 à 6 séances par semaine)
        data[f'nr. sessions{suffix}'] = np.random.randint(2, 7, n_rows)

        # Kilométrage total (ex: entre 20 et 100 km par semaine)
        data[f'total km{suffix}'] = np.random.uniform(20.0, 100.0, n_rows)

        # Séances de renforcement musculaire (ex: 0 à 3 séances)
        data[f'strength training{suffix}'] = np.random.randint(0, 4, n_rows)

        # Heures d'entraînement alternatif (ex: vélo, natation, entre 0 et 4 heures)
        data[f'hours alternative{suffix}'] = np.random.uniform(0.0, 4.0, n_rows)

    # 2. Ajout d'une colonne d'identification de l'athlète et de la semaine (optionnel mais utile)
    df = pd.DataFrame(data)
    df.insert(0, 'week_id', range(1, n_rows + 1))

    # 3. Simulation de la cible (injury_label) avec un taux de base d'environ 1.4% (0.014)
    # On ajoute un léger effet de risque basé sur une forte augmentation du kilométrage (simple rule) pour rendre les données un peu réalistes
    prev_2w_avg = (df['total km.1'] + df['total km.2']) / 2.0
    risk_factor = df['total km'] / prev_2w_avg

    # Probabilité de blessure de base + boost si le risque s'envole (ACWR élevé)
    base_prob = 0.014
    probabilities = base_prob * np.clip(risk_factor, 0.5, 3.0)
    probabilities = np.clip(probabilities, 0.001, 0.20) # Normalisation entre 0.1% et 20%

    # Tirage aléatoire bernoullien pour les étiquettes (0 ou 1)
    df['injury'] = [np.random.choice([0, 1], p=[1 - p, p]) for p in probabilities]

    return df

if __name__ == "__main__":
    # Génération du jeu de données (par exemple 2000 lignes)
    df_dataset = generate_injury_dataset(n_rows=2000)

    # Affichage des premières lignes et du taux de blessure réel
    print("Aperçu des premières colonnes du DataFrame généré :")
    print(df_dataset[['total km', 'total km.1', 'total km.2', 'injury']].head())

    actual_rate = df_dataset['injury'].mean()
    print(f"\nTaux de blessure effectif dans le fichier généré : {actual_rate:.4f} (~1.4% attendu)")

    # Sauvegarde au format CSV
    output_filename = "injury_prediction_dataset.csv"
    df_dataset.to_csv(output_filename, index=False)
    print(f"\nFichier CSV généré avec succès : '{output_filename}'")
