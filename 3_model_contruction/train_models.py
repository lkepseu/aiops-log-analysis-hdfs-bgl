# train_models.py

import argparse, os
from typing import Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from configs.build_chronological_matrix import build_chronological_matrix
from configs.data_utils import load_hdfs_matrix_and_labels
from random_forest_model import train_eval_random_forest_timeseries
from logistic_regression_model import train_eval_logistic_regression_timeseries
from configs.compute_permutation_importance import (
    compute_permutation_importance,
)

def train_model(
    dataset_name: str,
    matrix_csv: str,
    labels_csv: Optional[str] = None,
    structured_csv: Optional[str] = None,
    model_name: str = "both",
    n_splits: int = 5,
) -> None:
    """
    Entraîne RandomForest et/ou Logistic Regression avec une stratégie
    TimeSeries Cross-Validation (Rolling Window / Sliding-origin).

    - Respect strict de l'ordre temporel
    - Aucune fuite du futur vers le passé
    - Les données de test NE SONT PAS rebalancées
    - Seul le TRAIN utilise class_weight="balanced"
    """

    dataset = dataset_name.lower()

    # -----------------------------------------------------
    # 1) Construction de la matrice chronologique si BGL
    # -----------------------------------------------------
    final_matrix_csv = matrix_csv
    if dataset != "hdfs":
        if structured_csv is not None:
            root, ext = os.path.splitext(matrix_csv)
            output_csv = root + "_right_chrono" + ext

            print(f"[BGL] Construction de la matrice chronologique → {output_csv}")
            build_chronological_matrix(
                structured_csv=structured_csv,
                matrix_csv=matrix_csv,
                output_csv=output_csv,
                dataset=dataset_name
            )
            final_matrix_csv = output_csv
        else:
            print("[HDFS] Aucun structured_csv fourni, on suppose la matrice déjà chronologique.")

    # -----------------------------------------------------
    # 2) Chargement X, y (triés chronologiquement)
    # -----------------------------------------------------
    print("[INFO] Chargement des features et labels...")
    X, y = load_hdfs_matrix_and_labels(
        matrix_csv=final_matrix_csv,
        labels_csv=labels_csv,
        label_col="Label"
    )

    # Vérification du tri chronologique
    # (X doit avoir été construit avec un 'timestamp' trié)
    if "first_ts" in X.columns:
        X = X.sort_values("first_ts")  # optionnel selon ton pipeline

    print(f"[INFO] Taille dataset : {len(X)} lignes")

    # -----------------------------------------------------
    # 3) Entraînement via TimeSeries Cross-Validation
    # -----------------------------------------------------
    model = model_name.lower()

    if model in ("rf", "both"):
        print("\n[MODEL] RandomForest - TimeSeries Cross-Validation")
        train_eval_random_forest_timeseries(
            X=X,
            y=y,
            n_splits=n_splits
        )

    if model in ("lr", "both"):
        print("\n[MODEL] LogisticRegression - TimeSeries Cross-Validation")
        train_eval_logistic_regression_timeseries(
            X=X,
            y=y,
            n_splits=n_splits
        )

    if model not in ("rf", "lr", "both"):
        raise ValueError(f"model_name doit être 'rf', 'lr' ou 'both', reçu: {model_name}")

    # 4) Interprétation – Importance des features
    print("\n[INTERPRETATION] Importance des features (Permutation Importance, TimeSeries CV)")

    if model in ("rf", "both"):
        print(f"[INTERPRETATION] RandomForest – Permutation Importance ({dataset})")
        rf_base = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
            # pas de class_weight
        )
        compute_permutation_importance(
            model=rf_base,
            X=X,
            y=y,
            n_splits=n_splits,
            n_repeats=10,
            random_state=42,
        )

    if model in ("lr", "both"):
        print(f"[INTERPRETATION] LogisticRegression – Permutation Importance ({dataset})")
        lr_base = LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            random_state=42,
            # pas de class_weight
        )
        compute_permutation_importance(
            model=lr_base,
            X=X,
            y=y,
            n_splits=n_splits,
            n_repeats=10,
            random_state=42,
        )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RandomForest / LogisticRegression avec TimeSeries Cross-Validation."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Nom du dataset.")
    parser.add_argument("--matrix_csv", type=str, required=True, help="Chemin vers la matrice (HDFS ou BGL).")
    parser.add_argument("--labels_csv", type=str, required=False, help="Chemin vers le fichier de labels.")
    parser.add_argument("--structured_csv", type=str, required=False, help="Chemin vers le log structuré pour BGL.")
    parser.add_argument("--model", type=str, default="both", help="Modèle: 'rf', 'lr' ou 'both'.")
    parser.add_argument("--splits", type=int, default=5, help="Nombre de splits TimeSeries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(
        dataset_name=args.dataset,
        matrix_csv=args.matrix_csv,
        labels_csv=args.labels_csv,
        structured_csv=args.structured_csv,
        model_name=args.model,
        n_splits=args.splits,
    )


if __name__ == "__main__":
    main()
