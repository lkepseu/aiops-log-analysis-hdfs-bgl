import argparse
from typing import Literal

import pandas as pd

from build_bgl_matrix import build_bgl_matrix_sliding
from build_hdfs_matrix import build_hdfs_matrix

# Fonction pour générer et sauvegarder la matrice de features pour un dataset donné
def generate_features_matrix(
    dataset: Literal["bgl", "hdfs"],
    input_path: str,
    output_path: str,
    timestamp_col: str = "Timestamp",
    window_minutes: int = 5,
    step_minutes: int = 1,
) -> pd.DataFrame:

    # Étape 1. Charger le CSV structuré
    df = pd.read_csv(input_path)

    # Étape 2. Orienter vers le bon constructeur selon le dataset
    if dataset.lower() == "bgl":
        matrix = build_bgl_matrix_sliding(
            df=df,
            timestamp_col=timestamp_col,
            window_minutes=window_minutes,
            step_minutes=step_minutes,
        )

    elif dataset.lower() == "hdfs":
        # Construction de la matrice avec `build_hdfs_matrix(df: pd.DataFrame) -> pd.DataFrame`
        matrix = build_hdfs_matrix(df)

    else:
        raise ValueError(f"Dataset non supporté: {dataset}. Utilise 'bgl' ou 'hdfs'.")

    # Étape 3. Sauvegarder la matrice en CSV
    matrix.to_csv(output_path, index=False)

    return matrix


# Pour des tests rapides (cette fonction permet de parser les arguments en ligne de commande.)
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Génération de matrices de features pour BGL ou HDFS."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["bgl", "hdfs"],
        help="Nom du dataset à traiter (bgl ou hdfs).",
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Chemin vers le CSV structuré (Drain).",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Chemin de sortie pour la matrice (CSV).",
    )

    parser.add_argument(
        "--timestamp-col",
        type=str,
        default="Timestamp",
        help="Nom de la colonne timestamp (utilisé pour BGL).",
    )

    parser.add_argument(
        "--window-minutes",
        type=int,
        default=5,
        help="Taille de la fenêtre glissante en minutes (BGL, défaut=5).",
    )

    parser.add_argument(
        "--step-minutes",
        type=int,
        default=1,
        help="Pas de la fenêtre glissante en minutes (BGL, défaut=1).",
    )

    return parser.parse_args()


# Point d’entrée CLI
def main() -> None:
    args = _parse_args()

    generate_features_matrix(
        dataset=args.dataset,
        input_path=args.input,
        output_path=args.output,
        timestamp_col=args.timestamp_col,
        window_minutes=args.window_minutes,
        step_minutes=args.step_minutes,
    )


if __name__ == "__main__":
    main()