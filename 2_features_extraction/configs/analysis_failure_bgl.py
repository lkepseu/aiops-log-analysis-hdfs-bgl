"""
analysis_failure_bgl.py
-----------------------
Outils d'analyse du taux d'anomalies dans BGL à partir de la matrice
par fenêtres temporelles (sliding windows).

Fonctionnalités :
- identify_failure_event_ids             : Identification des EventId de failure
- plot_window_anomaly_count              : Histogramme des anomalies (count plot)
"""
import pandas as pd
import matplotlib.pyplot as plt

# Identifier les EventId associés à des logs de type "failure"
def identify_failure_event_ids(structured_csv: str) -> list[str]:
    import pandas as pd

    # Lecture du fichier structuré (Drain)
    df = pd.read_csv(structured_csv)

    # Vérifications minimales
    if "Label" not in df.columns:
        raise ValueError("Le CSV structuré doit contenir une colonne 'Label'.")
    if "EventId" not in df.columns:
        raise ValueError("Le CSV structuré doit contenir une colonne 'EventId'.")

    # Logs anormaux = Label différent de "-"
    df_failures = df[df["Label"] != "-"]

    # Extraire les EventId associés à ces lignes
    failure_event_ids = (
        df_failures["EventId"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return failure_event_ids

# Helper: permet de rajouter : - failure_count (nombre total d'événements de type failure dans la fenêtre)
# et : is_anomalous (booléen, True si la fenêtre contient au moins un failure)
def _add_window_anomaly_flags(
    df: pd.DataFrame,
    failure_event_ids: list[str],
) -> pd.DataFrame:
    df = df.copy()

    # Colonnes EventId = toutes les colonnes qui commencent par "E"
    event_cols = [c for c in df.columns if c.startswith("E")]

    # On limite aux EventId de failure
    failure_cols = [c for c in event_cols if c in failure_event_ids]
    if not failure_cols:
        raise ValueError("Aucune colonne EventId de failure trouvée dans la matrice.")

    # Nombre total de failures par fenêtre
    df["failure_count"] = df[failure_cols].sum(axis=1)

    # Fenêtre anormale si au moins un failure
    df["is_anomalous"] = df["failure_count"] > 0

    return df



# Plot. Un count plot du nombre de fenêtres normales vs anormales
def plot_window_anomaly_count(
    matrix_csv: str,
    failure_event_ids: list[str],
    title: str = "BGL — Histogramme des fenêtres normales vs anormales",
) -> None:
    df = pd.read_csv(matrix_csv)

    # Ajout des labels anomalies
    df = _add_window_anomaly_flags(df, failure_event_ids)

    counts = df["is_anomalous"].value_counts().sort_index()

    # Mapping simple
    labels = ["normal", "anomalous"]
    values = [counts.get(False, 0), counts.get(True, 0)]

    # === Plot ===
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.ylabel("Nombre de fenêtres")
    plt.title(title)

    # === Affichage console ===
    total = sum(values)
    ratio_anomalies = values[1] / total if total > 0 else 0.0

    print(f"[INFO] Total fenêtres : {total}")
    print(f"[INFO] Fenêtres anormales : {values[1]} "
          f"({ratio_anomalies:.4f} ≈ {ratio_anomalies*100:.2f}%)")

    # === Sauvegarde du CSV avec labels ===
    output_csv = matrix_csv.replace(".csv", "_with_labels.csv")

    # 1) Supprimer failure_count si présent
    if "failure_count" in df.columns:
        df = df.drop(columns=["failure_count"])

    # 2) Convertir is_anomalous → Label
    if "is_anomalous" not in df.columns:
        raise ValueError("La colonne 'is_anomalous' est absente du dataframe.")

    df["Label"] = df["is_anomalous"].map({False: 0, True: 1})

    # 3) Supprimer l’ancienne colonne is_anomalous
    df = df.drop(columns=["is_anomalous"])

    # 4) Sauvegarde finale
    df.to_csv(output_csv, index=False)

    print(f"[INFO] Fichier généré : {output_csv}")
    print(f"[INFO] Colonnes finales : {list(df.columns)}")
    print(df.head(10))

    plt.tight_layout()
    plt.show()
