"""
windows.py
----------
Module regroupant les primitives génériques de fenêtrage et de découpage
utilisées pour l’extraction de features pour les logs.

Fonctions :
- generate_time_windows     : Génération d'intervalles temporels successifs (sliding windows).
- apply_sliding_window      : Application d’un fenêtrage temporel à un DataFrame (ex. BGL).
- apply_windows_by_session  : Découpage par identifiant de session logique (ex. BlockId pour HDFS).
"""
import pandas as pd
from typing import Iterator, Tuple, Callable, Optional


def generate_time_windows(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    window_minutes: int,
    step_minutes: int,
) -> Iterator[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Génère des intervalles temporels successifs de type sliding window.

    Exemple :
        start_time = 08:00
        window_minutes = 5
        step_minutes = 1
        => fenêtres :
            08:00 → 08:05
            08:01 → 08:06
            08:02 → 08:07
            ...

    Paramètres
    ----------
    start_time : pd.Timestamp
    end_time : pd.Timestamp
    window_minutes : int
    step_minutes : int

    Retour
    ------
    Iterator[(start, end)]
        Pour itérer fenêtre par fenêtre.
    """

    window_delta = pd.Timedelta(minutes=window_minutes)
    step_delta = pd.Timedelta(minutes=step_minutes)

    current_start = start_time

    # Génération successive des fenêtres tant que start < end_time
    while current_start <= end_time:
        current_end = current_start + window_delta
        yield current_start, current_end # Renvoie cette fenêtre puis continue
        current_start += step_delta


def apply_sliding_window(
    df: pd.DataFrame,
    timestamp_col: str,
    window_minutes: int,
    step_minutes: int,
    agg_func,
) -> pd.DataFrame:
    """
    Applique un fenêtrage glissant générique à un DataFrame.

    `agg_func` transforme chaque sous-DataFrame (une fenêtre) → en un dict de features :
        agg_func(df_window, window_start, window_end) → dict

    Retour
    ------
    pd.DataFrame
        Matrice finale → une ligne par fenêtre.
    """

    # Étape 1. Nettoyer et ordonner les timestamps
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], format="%Y-%m-%d-%H.%M.%S.%f", errors="coerce")
    df = df.dropna(subset=[timestamp_col])
    df = df.sort_values(timestamp_col)

    # Étape 2. Définir les limites temporelles
    t_min = df[timestamp_col].min()
    t_max = df[timestamp_col].max()

    # Étape 3. Générer les fenêtres avec generate_time_windows()
    rows = []
    for w_start, w_end in generate_time_windows(
        t_min, t_max, window_minutes, step_minutes
    ):
        # Étape 4. Extraire les logs dans la fenêtre
        mask = (df[timestamp_col] >= w_start) & (df[timestamp_col] < w_end) # Garde seulement les lignes dans la fenêtre de temps
        df_window = df[mask]

        # Étape 5. Extraire les features via agg_func
        row = agg_func(df_window, w_start, w_end)

        rows.append(row)

    # Étape 6. Transformer toutes les lignes en DataFrame
    return pd.DataFrame(rows)


def apply_windows_by_session(
    df: pd.DataFrame,
    session_extractor: Callable[[pd.Series], Optional[str]],
    agg_func: Callable[[pd.DataFrame, str], dict],
    session_col: str = "SessionId",
) -> pd.DataFrame:
    """
    Applique un fenêtrage par session (Block pour HDFS)
    en utilisant une fonction générique pour extraire l'ID de session
    à partir de chaque ligne.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame de logs.

    session_extractor : callable
        Fonction appelée pour chaque ligne :
            session_extractor(row: pd.Series) -> str | None
            - pour HDFS : extraire 'blk_xxx' depuis la colonne 'Content'

    agg_func : callable
        Fonction qui agrège les logs d'une session en un dict de features :
            agg_func(df_session: pd.DataFrame, session_id: str) -> dict

    session_col : str
        Nom de la colonne qui sera créée pour stocker l'ID de session.

    Retour
    ------
    pd.DataFrame
        Matrice où chaque ligne = une session, colonnes = features.
    """

    # Étape 1. Créer une copie pour ne pas modifier le DataFrame d'origine
    df = df.copy()

    # Étape 2. Appliquer le session_extractor sur chaque ligne
    #          → création d'une nouvelle colonne session_col
    df[session_col] = df.apply(session_extractor, axis=1)

    # Étape 3. Supprimer les lignes sans session_id (None / NaN)
    df = df.dropna(subset=[session_col])

    # Étape 4. Regrouper les logs par session_id
    grouped = df.groupby(session_col)

    # Étape 5. Appliquer l'agg_func sur chaque groupe de session
    rows = []
    for session_id, df_session in grouped:
        row = agg_func(df_session, session_id)
        rows.append(row)

    # Étape 6. Transformer la liste de dicts en DataFrame
    matrix = pd.DataFrame(rows)

    # Étape 7. Remplacer les valeurs manquantes par 0
    # (uniquement sur les colonnes numériques pour garder les ID/textes intacts)
    numeric_cols = matrix.select_dtypes(include=["number", "float", "int"]).columns
    matrix[numeric_cols] = matrix[numeric_cols].fillna(0)

    # Étape 8. Convertir les colonnes numériques en entiers
    # (utile pour les histogrammes d'EventId)
    matrix[numeric_cols] = matrix[numeric_cols].astype(int)

    return matrix
