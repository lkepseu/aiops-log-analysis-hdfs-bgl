# train_models.py

import argparse, os
from typing import Optional

import pandas as pd  # seulement si tu en as besoin ailleurs, sinon tu peux enlever

from configs.build_chronological_matrix import build_chronological_matrix
from configs.data_utils import load_hdfs_matrix_and_labels, split_60_20_20_chrono
from random_forest_model import train_eval_random_forest
from logistic_regression_model import train_eval_logistic_regression

def train_model(
    dataset_name: str,
    matrix_csv: str,
    labels_csv: str,
    structured_csv: Optional[str] = None,
    model_name: str = "both",
) -> None:
    """
    Entraîne les modèles (RandomForest et/ou Logistic Regression) sur un dataset.
    """
    dataset = dataset_name.lower()

    # 1) Construire la matrice chronologique si besoin
    final_matrix_csv = matrix_csv
    if dataset != "hdfs":
        if structured_csv is not None:
            root, ext = os.path.splitext(matrix_csv)
            output_csv = root + "_right_chrono" + ext
            print(f"[BGL] Construction de la matrice chronologique → {output_csv}")
            build_chronological_matrix(structured_csv=structured_csv, matrix_csv=matrix_csv, output_csv=output_csv, dataset=dataset_name)
            final_matrix_csv = output_csv
        else:
            print("[HDFS] Aucun structured_csv fourni, on suppose la matrice déjà dans le bon ordre (ou avec first_ts).")

    # 2) Charger X, y
    print("[INFO] Chargement des features et labels...")
    X, y = load_hdfs_matrix_and_labels(matrix_csv=final_matrix_csv, labels_csv=labels_csv, label_col="Label")

    # 3) Split 60/20/20 chronologique
    print("[INFO] Split 60/20/20 chronologique...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_60_20_20_chrono(X, y)
    print(f"[INFO] Train: {len(X_train)}  |  Val: {len(X_val)}  |  Test: {len(X_test)}")

    # 4) Entraînement / évaluation des modèles selon model_name
    model = model_name.lower()
    if model in ("rf", "both"):
        print("\n[MODEL] RandomForest for {dataset_name}")
        train_eval_random_forest(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

    if model in ("lr", "both"):
        print("\n[MODEL] LogisticRegression for {dataset_name}")
        train_eval_logistic_regression(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

    if model not in ("rf", "lr", "both"):
        raise ValueError(f"model_name doit être 'rf', 'lr' ou 'both', reçu: {model_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RandomForest / LogisticRegression avec split 60/20/20 chrono.")
    parser.add_argument("--dataset", type=str, required=True, help="Nom du dataset.")
    parser.add_argument("--matrix_csv", type=str, required=True, help="Chemin vers la matrice HDFS (BlockId x EventId).")
    parser.add_argument("--labels_csv", type=str, required=False, help="Chemin vers le fichier de labels (BlockId, Label).")
    parser.add_argument("--structured_csv", type=str, required=False, help="Chemin vers le log structuré HDFS (optionnel).")
    parser.add_argument("--model", type=str, required=False, default="both", help="Modèle à entraîner: 'rf', 'lr' ou 'both'.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(
        dataset_name=args.dataset,
        matrix_csv=args.matrix_csv,
        labels_csv=args.labels_csv,
        structured_csv=args.structured_csv,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
