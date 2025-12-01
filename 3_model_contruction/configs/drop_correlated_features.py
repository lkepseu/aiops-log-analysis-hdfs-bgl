from __future__ import annotations
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ne garde que les colonnes numériques (float, int).
    Copie le DataFrame pour éviter les effets de bord.
    """
    return df.select_dtypes(include=[np.number]).copy()


def _choose_feature_to_drop(
    feat_a: str,
    feat_b: str,
    variances: pd.Series,
    importances: Optional[Dict[str, float]] = None,
) -> str:
    """
    Décide quelle feature supprimer entre feat_a et feat_b.

    Règles :
      - si des importances sont fournies : on garde la feature la plus importante
      - sinon : on garde la feature avec la plus grande variance
      → la feature RETOURNÉE est celle à SUPPRIMER.
    """
    if importances is not None:
        imp_a = importances.get(feat_a, 0.0)
        imp_b = importances.get(feat_b, 0.0)
        return feat_a if imp_a < imp_b else feat_b

    var_a = variances.get(feat_a, 0.0)
    var_b = variances.get(feat_b, 0.0)
    return feat_a if var_a < var_b else feat_b


def drop_correlated_features(
    df: pd.DataFrame,
    corr_threshold: float = 0.8,
    feature_importances: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Supprime les features fortement corrélées (|corr| >= corr_threshold).

    - Ne travaille que sur les colonnes numériques.
    - Si feature_importances est fourni, il est utilisé pour décider
      quelle feature supprimer dans une paire corrélée.
    - Sinon, la variance est utilisée comme critère.

    Retourne un NOUVEAU DataFrame.
    """
    # Séparer numérique / non numérique
    numeric_df = _numeric_frame(df)
    non_numeric_df = df.drop(columns=numeric_df.columns, errors="ignore")
    # --- Protection supplémentaire : retirer Label du numérique ---
    if "Label" in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=["Label"])

    if numeric_df.shape[1] <= 1:
        return numeric_df

    corr_matrix = numeric_df.corr(method="pearson").abs()
    variances = numeric_df.var()

    # Partie supérieure de la matrice de corrélation
    mask_upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    upper = corr_matrix.where(mask_upper)

    to_drop: set[str] = set()

    print(f"\n=== [CORR] Début filtrage corrélation (seuil = {corr_threshold}) ===")
    initial_cols = list(numeric_df.columns)

    for col in upper.columns:
        # lignes fortement corrélées à `col`
        high_corr_rows: Iterable[str] = upper.index[upper[col] >= corr_threshold]
        for row in high_corr_rows:
            if row in to_drop or col in to_drop:
                continue

            drop_feat = _choose_feature_to_drop(
                feat_a=row,
                feat_b=col,
                variances=variances,
                importances=feature_importances,
            )
            to_drop.add(drop_feat)
            print(f"[CORR] Suppression: '{drop_feat}' (corrélation élevée avec '{col}' et '{row}')")

    print(f"[CORR] Total supprimées: {len(to_drop)} / {len(initial_cols)}")
    print("==========================================")

    numeric_reduced = numeric_df.drop(columns=list(to_drop), errors="ignore")

    # Remettre Label dans les colonnes non numériques pour conservation finale
    if "Label" in df.columns and "Label" not in non_numeric_df.columns:
        non_numeric_df["Label"] = df["Label"]

    # Réattacher les colonnes non numériques (BlockId etc.)
    df_final = pd.concat([non_numeric_df, numeric_reduced], axis=1)

    # Optionnel : respecter l'ordre original des colonnes autant que possible
    # en réordonnant selon df.columns
    ordered_cols = [c for c in df.columns if c in df_final.columns]
    ordered_cols += [c for c in df_final.columns if c not in ordered_cols]

    # === Affichage debug avant return ===
    print("\n[DEBUG] Aperçu du DataFrame final après filtrage (10 premières lignes) :")
    print(df_final.head(10))
    print(f"[DEBUG] Nombre total de colonnes finales : {len(df_final.columns)}")
    print(f"[DEBUG] Liste des colonnes finales : {list(df_final.columns)}\n")

    return df_final[ordered_cols]