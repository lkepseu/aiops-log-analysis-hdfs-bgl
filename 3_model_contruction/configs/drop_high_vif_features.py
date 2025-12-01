from __future__ import annotations

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from numpy.linalg import LinAlgError


def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ne garde que les colonnes numériques (float, int).
    Copie le DataFrame pour éviter les effets de bord.
    """
    return df.select_dtypes(include=[np.number]).copy()


# def _compute_vif_series(numeric_features: pd.DataFrame) -> pd.Series:
#     """
#     Calcule le VIF pour chaque colonne numérique.
#     """
#     if numeric_features.shape[1] == 0:
#         return pd.Series(dtype=float)
#
#     values = numeric_features.values
#     vifs = [
#         variance_inflation_factor(values, i)
#         for i in range(values.shape[1])
#     ]
#     return pd.Series(vifs, index=numeric_features.columns, name="VIF")





def _compute_vif_series(X: pd.DataFrame) -> pd.Series:
    """
    Calcule le VIF pour chaque colonne de X en étant robuste :
    - supprime les colonnes constantes avant le calcul
    - en cas de problème numérique (SVD non convergente), renvoie VIF = inf
    """
    if X.shape[1] == 0:
        return pd.Series(dtype=float)

    # 1) Supprimer les colonnes constantes (variance = 0)
    variances = X.var(axis=0)
    non_constant_cols = variances[variances > 0].index.tolist()
    X_nc = X[non_constant_cols]

    if X_nc.shape[1] == 0:
        # Toutes les colonnes étaient constantes
        return pd.Series(
            data=[],
            index=[],
            dtype=float,
            name="VIF"
        )

    values = np.asarray(X_nc.values, dtype=float)

    vifs = []
    for i in range(values.shape[1]):
        try:
            vif_val = variance_inflation_factor(values, i)
        except LinAlgError:
            # SVD non convergente -> multicolinéarité extrême -> on marque VIF = inf
            vif_val = float("inf")
        except Exception:
            # Par sécurité, toute autre erreur -> on marque VIF = inf
            vif_val = float("inf")

        vifs.append(vif_val)

    vif_series = pd.Series(vifs, index=X_nc.columns, name="VIF")
    return vif_series


def drop_high_vif_features(
    df: pd.DataFrame,
    vif_threshold: float = 10.0,
) -> pd.DataFrame:
    """
    Supprime récursivement les features avec VIF > vif_threshold
    parmi les colonnes NUMÉRIQUES uniquement.
    Les colonnes non numériques (comme BlockId) sont préservées.
    """

    # Séparer numérique / non numérique
    numeric_df = _numeric_frame(df)
    non_numeric_df = df.drop(columns=numeric_df.columns, errors="ignore")
    # --- Protection supplémentaire : retirer Label du numérique ---
    if "Label" in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=["Label"])

    # Rien à filtrer
    if numeric_df.shape[1] <= 1:
        print("[VIF] Moins de 2 colonnes numériques, rien à filtrer.")
        return pd.concat([non_numeric_df, numeric_df], axis=1)

    print(f"\n=== [VIF] Début filtrage VIF (seuil = {vif_threshold}) ===")

    initial_cols = list(numeric_df.columns)
    removed = 0

    # Copie de travail :
    working_df = numeric_df.copy()

    # Boucle récursive
    while working_df.shape[1] > 1:
        vif_values = _compute_vif_series(working_df)
        max_vif = vif_values.max()

        if max_vif <= vif_threshold:
            break

        to_drop = vif_values.idxmax()
        removed += 1
        print(f"[VIF] Suppression: '{to_drop}' (VIF = {max_vif:.2f})")

        working_df = working_df.drop(columns=[to_drop])

    print(f"[VIF] Total supprimées: {removed} / {len(initial_cols)}")
    print("==========================================")

    numeric_reduced = working_df
    # Remettre Label dans les colonnes non numériques pour conservation finale
    if "Label" in df.columns and "Label" not in non_numeric_df.columns:
        non_numeric_df["Label"] = df["Label"]

    # Réattacher les colonnes non numériques
    df_final = pd.concat([non_numeric_df, numeric_reduced], axis=1)

    # Restaurer au mieux l'ordre d'origine
    ordered_cols = [c for c in df.columns if c in df_final.columns]
    ordered_cols += [c for c in df_final.columns if c not in ordered_cols]

    # === Affichage debug avant return ===
    print("\n[DEBUG] Aperçu du DataFrame final après filtrage (10 premières lignes) :")
    print(df_final.head(10))
    print(f"[DEBUG] Nombre total de colonnes finales : {len(df_final.columns)}")
    print(f"[DEBUG] Liste des colonnes finales : {list(df_final.columns)}\n")

    return df_final[ordered_cols]
