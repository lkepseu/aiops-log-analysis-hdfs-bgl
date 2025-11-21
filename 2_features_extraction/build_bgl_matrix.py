import pandas as pd
from typing import Dict, Any
from configs.windows import apply_sliding_window


# Étape 1. Fonction d’agrégation spécifique à BGL (histogramme EventId)
def bgl_agg_eventid_histogram(
    df_window: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp
) -> Dict[str, Any]:

    # Étape A. Compter les occurrences des EventId
    event_counts = df_window["EventId"].value_counts().to_dict()

    # Étape B. Ajouter les métadonnées temporelles
    row = {
        "window_start": window_start,
        "window_end": window_end,
        **event_counts
    }

    return row


# Étape 2. Fonction principale : construction de la matrice BGL
def build_bgl_matrix_sliding(
    df: pd.DataFrame,
    timestamp_col: str = "Timestamp",
    window_minutes: int = 5,
    step_minutes: int = 1,
) -> pd.DataFrame:

    # Étape A. Utiliser la primitive générique apply_sliding_window()
    matrix = apply_sliding_window(
        df=df,
        timestamp_col=timestamp_col,
        window_minutes=window_minutes,
        step_minutes=step_minutes,
        agg_func=bgl_agg_eventid_histogram,
    )

    # Étape B. Remplacer les valeurs manquantes par 0
    matrix = matrix.fillna(0)

    # Étape C. Tri des colonnes (EventId triés afin d'avoir le format : Start, end, E1, E2, ..., EN)
    event_cols = [c for c in matrix.columns if c not in ("window_start", "window_end")]
    matrix = matrix[["window_start", "window_end"] + sorted(event_cols)]

    return matrix
