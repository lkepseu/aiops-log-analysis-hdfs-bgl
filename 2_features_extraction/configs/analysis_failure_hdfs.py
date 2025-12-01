import pandas as pd
import matplotlib.pyplot as plt
import os

# Charge la matrice HDFS et retourne df_normaux, df_corrompus, event_cols
def load_and_split_hdfs_matrix(
    matrix_csv: str,
    labels_csv: str,
    output_csv: str = None,
    label_col: str = "Label"
):
    """
    Charge la matrice HDFS, ajoute la colonne Label fusionnée depuis labels.csv,
    convertit Normal/Anomaly en 0/1, sauvegarde un fichier enrichi, puis
    renvoie df_normaux, df_corrompus, event_cols.

    Paramètres
    ----------
    matrix_csv : str
        Chemin vers HDFS*_matrix.csv
    labels_csv : str
        Chemin vers HDFS*_label.csv
    output_csv : str | None
        Chemin du csv final (avec colonne Label ajoutée).
        Si None, génère automatiquement "HDFS_matrix_with_labels.csv"
    label_col : str
        Nom de la colonne label dans labels_csv ("Label")
    """

    # --------------------
    # 1. Charger les données
    # --------------------
    df_matrix = pd.read_csv(matrix_csv)
    df_labels = pd.read_csv(labels_csv)

    if "BlockId" not in df_matrix.columns:
        raise ValueError("La matrice doit contenir une colonne 'BlockId'.")

    if "BlockId" not in df_labels.columns or label_col not in df_labels.columns:
        raise ValueError("labels_csv doit contenir 'BlockId' et 'Label'.")

    # --------------------
    # 2. Convertir Normal / Anomaly en 0 / 1
    # --------------------
    df_labels = df_labels.copy()
    df_labels[label_col] = df_labels[label_col].map({
        "Normal": 0,
        "Anomaly": 1
    })

    # Sécurité : si d’autres valeurs apparaissent
    if df_labels[label_col].isna().any():
        raise ValueError("Valeurs inattendues dans Label : seulement 'Normal' et 'Anomaly' autorisés.")

    # --------------------
    # 3. Fusion matrice + labels via BlockId
    # --------------------
    df_merged = df_matrix.merge(df_labels[["BlockId", label_col]], on="BlockId", how="left")

    # Vérification : aucun BlockId perdu
    if df_merged[label_col].isna().any():
        missing = df_merged[df_merged[label_col].isna()]["BlockId"].unique()
        raise ValueError(f"Certains BlockId n'ont pas de Label : {missing[:10]} ...")

    # --------------------
    # 4. Sauvegarder la matrice enrichie
    # --------------------
    if output_csv is None:
        dirname = os.path.dirname(matrix_csv)
        output_csv = os.path.join(dirname, "HDFS_matrix_with_labels.csv")

    df_merged.to_csv(output_csv, index=False)
    print(f"[INFO] Fichier enrichi sauvegardé : {output_csv}")

    # --------------------
    # 5. Définir les EventIds (colonnes E1, E2, ...)
    # --------------------
    event_cols = [c for c in df_matrix.columns if c.startswith("E")]
    if not event_cols:
        raise ValueError("Aucune colonne 'E*' trouvée dans la matrice.")

    # --------------------
    # 6. Séparer Normal vs Corrompu
    # --------------------
    df_normaux = df_merged[df_merged[label_col] == 0]
    df_corrompus = df_merged[df_merged[label_col] == 1]

    return df_normaux, df_corrompus, event_cols



# Etape 1. Histogram — EventId dans les blocks corrompus
def plot_eventid_histogram_corrupted(df_corrompus, event_cols, top_k=20):
    totals = df_corrompus[event_cols].sum().sort_values(ascending=False)
    top = totals.head(top_k)

    plt.figure(figsize=(10, 5))
    plt.bar(top.index, top.values, color="red")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Occurrences")
    plt.xlabel("EventId")
    plt.title(f"HDFS — EventId dans les blocks corrompus (Top {top_k})")
    plt.tight_layout()
    plt.show()


# Etape 2. Histogram — EventId dans les blocks corrompus
def plot_eventid_histogram_normal(df_normaux, event_cols, top_k=20):
    totals = df_normaux[event_cols].sum().sort_values(ascending=False)
    top = totals.head(top_k)

    plt.figure(figsize=(10, 5))
    plt.bar(top.index, top.values, color="green")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Occurrences")
    plt.xlabel("EventId")
    plt.title(f"HDFS — EventId dans les blocks normaux (Top {top_k})")
    plt.tight_layout()
    plt.show()


# Etape 3. Barplot comparatif normal vs corrompu
def plot_eventid_normal_vs_corrupted(df_normaux, df_corrompus, event_cols, top_k=20):
    # Somme globale par EventId
    total_norm = df_normaux[event_cols].sum()
    total_corr = df_corrompus[event_cols].sum()

    # On garde les EventId les plus fréquents dans corrompu
    event_ids_sorted = total_corr.sort_values(ascending=False).head(top_k).index

    data = pd.DataFrame({
        "normal": total_norm[event_ids_sorted],
        "corrupted": total_corr[event_ids_sorted],
    })

    # Barplot côte à côte
    plt.figure(figsize=(12, 5))
    bar_width = 0.4
    x = range(len(event_ids_sorted))

    plt.bar([v - bar_width/2 for v in x],
            data["normal"], width=bar_width, label="Normal", color="green")

    plt.bar([v + bar_width/2 for v in x],
            data["corrupted"], width=bar_width, label="Corrompu", color="red")

    plt.xticks(ticks=x, labels=event_ids_sorted, rotation=45, ha="right")
    plt.ylabel("Occurrences")
    plt.xlabel("EventId")
    plt.title(f"HDFS — Comparaison Normal vs Corrompu (Top {top_k})")
    plt.legend()
    plt.tight_layout()
    plt.show()



# Barplot simple montrant : le nombre de blocks normaux et le nombre de blocks corrompus
def plot_blockid_normal_vs_corrupted_ratio(
    labels_csv: str,
    label_col: str = "Label",
    title: str = "HDFS — Répartition des BlockId normaux vs corrompus"
) -> None:

    df = pd.read_csv(labels_csv)

    if label_col not in df.columns:
        raise ValueError(f"Le fichier labels doit contenir une colonne '{label_col}'.")

    # Comptage simple
    counts = df[label_col].value_counts().sort_index()  # 0 puis 1

    nb_normal = counts.get("Normal", 0)
    nb_corrupted = counts.get("Anomaly", 0)
    total = nb_normal + nb_corrupted
    ratio = nb_corrupted / total if total > 0 else 0

    print(f"[INFO] Blocks normaux   : {nb_normal}")
    print(f"[INFO] Blocks corrompus : {nb_corrupted}")
    print(f"[INFO] Taux d'anomalies : {ratio*100:.2f}%")

    # Bar plot
    labels = ["Normal (0)", "Corrompu (1)"]
    values = [nb_normal, nb_corrupted]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=["green", "red"], alpha=0.7)
    plt.ylabel("Nombre de blocks")
    plt.title(title)
    plt.tight_layout()
    plt.show()

