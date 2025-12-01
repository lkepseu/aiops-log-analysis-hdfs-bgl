# data_utils.py

from typing import List, Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split


def load_hdfs_matrix_and_labels(
    matrix_csv: str,
    labels_csv: Optional[str] = None,
    label_col: str = "Label",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Charge la matrice des features, fusionne avec les labels si nécessaire,
    trie chronologiquement, et renvoie X (features) et y (labels).
    """

    # --- Charger la matrice ---
    df_matrix = pd.read_csv(matrix_csv)
    df = df_matrix  # nom local simplifié

    # --- CAS 1 : le label existe déjà dans la matrice => on ignore labels_csv ---
    if label_col in df.columns:
        print(f"[INFO] La matrice contient déjà '{label_col}', labels_csv ignoré.")

    # --- CAS 2 : label absent => on fusionne avec labels_csv ---
    else:
        if labels_csv is None:
            raise ValueError(
                f"'{label_col}' absent de la matrice et aucun fichier labels_csv fourni."
            )

        df_labels = pd.read_csv(labels_csv)

        if label_col not in df_labels.columns:
            raise ValueError(
                f"Erreur : la colonne '{label_col}' est absente de {labels_csv}. "
                f"Colonnes disponibles : {list(df_labels.columns)}"
            )

        if "BlockId" not in df.columns or "BlockId" not in df_labels.columns:
            raise ValueError("La fusion requiert une colonne BlockId dans les deux CSV.")

        df = df.merge(df_labels[["BlockId", label_col]], on="BlockId", how="inner")
        print(f"[INFO] Fusion BlockId OK — df.shape = {df.shape}")

    # --- Tri chronologique si disponible ---
    for ts_col in ("first_ts", "window_start"):
        if ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
            df = df.sort_values(ts_col)
            print(f"[INFO] Tri chronologique appliqué sur '{ts_col}'.")
            break

    # --- Séparer X (features) / y (labels) ---
    exclude_cols = {"BlockId", label_col, "first_ts", "window_start", "window_end", "Label"}

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols]
    y = df[label_col]

    # --- Conversion des labels texte en 0/1 ---
    if y.dtype == object:
        y = y.map({
            "Normal": 0,
            "normal": 0,
            "Anomaly": 1,
            "anomaly": 1,
            "False": 0,
            "True": 1,
        })
    if y.dtype == bool:
        y = y.astype(int)

    if y.isna().any():
        raise ValueError(f"[ERROR] Labels invalides : {df[label_col].unique()}")

    print(f"[INFO] Features: {len(feature_cols)}  —  Labels: {y.value_counts().to_dict()}")

    # --- CHECK : affichage X + y ---
    print("\n[CHECK] Aperçu X + y (5 premières lignes) :")
    print(pd.concat([X, y], axis=1).head())

    # --- CHECK : corrélation des features avec le label ---
    print("\n[CHECK] Corrélation des features avec y :")
    df_corr = pd.concat([X, y], axis=1).corr()[label_col].sort_values(ascending=False)
    print(df_corr)

    # --- CHECK : features suspects (corrélation trop forte) ---
    suspicious = df_corr[abs(df_corr) > 0.8]  # seuil configurable
    if len(suspicious) > 1:  # y corrélé à y → 1 élément à ignorer
        print("\n[ALERT] Features avec corrélation > 0.8 (risque de leakage) :")
        print(suspicious)
    else:
        print("\n[INFO] Aucune corrélation anormale détectée — dataset sain.")

    return X, y


def split_60_20_20_chrono(
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Découpe les données en 60% / 20% / 20% (train/val/test) en conservant l'ordre actuel
    (qui doit déjà être chronologique).

    Paramètres
    ----------
    X : pd.DataFrame
        Features triées chronologiquement.
    y : pd.Series
        Labels correspondants.

    Retour
    ------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    n = len(X)
    if n != len(y):
        raise ValueError("X et y doivent avoir la même longueur.")

    idx_60 = int(0.6 * n)
    idx_80 = int(0.8 * n)

    X_train = X.iloc[:idx_60]
    y_train = y.iloc[:idx_60]

    X_val = X.iloc[idx_60:idx_80]
    y_val = y.iloc[idx_60:idx_80]

    X_test = X.iloc[idx_80:]
    y_test = y.iloc[idx_80:]

    return X_train, X_val, X_test, y_train, y_val, y_test

