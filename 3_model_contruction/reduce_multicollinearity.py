from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import argparse

import pandas as pd

from configs.drop_correlated_features import drop_correlated_features
from configs.drop_high_vif_features import drop_high_vif_features


def reduce_multicollinearity(
    dataset_name: str,
    df: pd.DataFrame,
    corr_threshold: float = 0.8,
    vif_threshold: float = 10.0,
    feature_importances: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Pipeline compact pour réduire la multicolinéarité :

      1) suppression des corrélations fortes (Pearson)
      2) suppression récursive des features avec VIF élevé

    Returns:
        DataFrame avec uniquement les colonnes numériques filtrées.
    """

    matrix_after_corr = drop_correlated_features(df=df, corr_threshold=corr_threshold, feature_importances=feature_importances)
    matrix_after_vif = matrix_after_corr
    if dataset_name != "bgl":
        matrix_after_vif = drop_high_vif_features(df=matrix_after_corr,vif_threshold=vif_threshold)

    return matrix_after_vif


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Réduction de multicolinéarité sur un CSV (corrélation + VIF).")

    parser.add_argument("--dataset", type=str, required=True, help="Nom du dataset.")
    parser.add_argument("--input", type=str, required=True, help="Chemin CSV d'entrée.")
    parser.add_argument("--output", type=str, required=True, help="Chemin CSV de sortie.")
    parser.add_argument("--corr", type=float, default=0.8, help="Seuil |corr| pour filtrage.")
    parser.add_argument("--vif", type=float, default=10.0, help="Seuil VIF pour filtrage.")
    parser.add_argument("--importances", type=str, default=None, help="CSV des importances (feature,importance).")

    args = parser.parse_args()

    # Chargement du fichier d'entrée
    df = pd.read_csv(args.input)

    # Importances optionnelles
    feature_importances = None
    if args.importances is not None:
        imp_df = pd.read_csv(args.importances)
        feature_importances = dict(zip(imp_df["feature"], imp_df["importance"]))

    # Exécution du pipeline
    df_reduced = reduce_multicollinearity(
        dataset_name=args.dataset,
        df=df,
        corr_threshold=args.corr,
        vif_threshold=args.vif,
        feature_importances=feature_importances,
    )

    df_reduced.to_csv(args.output, index=False)
    print(f"[OK] Fichier sauvegardé : {args.output}")