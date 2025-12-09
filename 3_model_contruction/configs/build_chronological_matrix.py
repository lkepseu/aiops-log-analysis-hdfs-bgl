import pandas as pd

def load_structured_logs(structured_csv: str) -> pd.DataFrame:
    """
    Charge le fichier structuré HDFS et construit une colonne timestamp réelle.
    Format attendu : LineId,Date,Time,Pid,Level,Component,Content,EventId,...
    """
    df = pd.read_csv(structured_csv)

    # Combiner Date + Time en un vrai timestamp
    # Exemple HDFS : Date = 81109  (08-11 09 ??)
    # Tu dois convertir Date/Time → timestamp ci-dessous :
    df["Date"] = df["Date"].astype(str)
    df["Time"] = df["Time"].astype(str).str.zfill(6)  # ex: "203615"

    # Transformation en HH:MM:SS
    df["TimeFmt"] = df["Time"].str.slice(0,2) + ":" + df["Time"].str.slice(2,4) + ":" + df["Time"].str.slice(4,6)

    # Transformation en date HDFS
    # Date = MMDD hhmmss (LogHub format)
    # ex: 81109 → "08-11"
    df["Month"] = df["Date"].str.slice(0,1)
    df["Day"]   = df["Date"].str.slice(1,3)

    # Construire un timestamp complet (année arbitraire mais cohérente)
    df["timestamp"] = pd.to_datetime(
        "2018-" + df["Month"] + "-" + df["Day"] + " " + df["TimeFmt"],
        errors="coerce"
    )

    return df


def compute_block_first_timestamp(df_struct: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque BlockId, on prend le timestamp de la première occurrence.
    """
    df = df_struct.copy()

    # Extraire BlockId depuis ParameterList quand EventId contient 'blk'
    df["BlockId"] = df["Content"].str.extract(r"(blk_[\-0-9]+)")

    # Enlever les lignes sans block
    df = df.dropna(subset=["BlockId"])

    # Timestamp de première apparition
    df_block_ts = (
        df.groupby("BlockId")["timestamp"]
          .min()
          .reset_index()
          .rename(columns={"timestamp": "first_ts"})
    )

    return df_block_ts


def reorder_matrix_by_chronology(matrix_csv: str, block_ts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fusionne la matrice EventId-block avec les timestamps
    puis trie les blocks par ordre chronologique.
    """
    df_matrix = pd.read_csv(matrix_csv)

    # Fusion
    df = df_matrix.merge(block_ts_df, on="BlockId", how="left")

    # Trier par timestamp
    df = df.sort_values("first_ts")

    return df



def build_chronological_matrix(
    structured_csv: str,
    matrix_csv: str,
    output_csv: str,
    dataset: str,
) -> None:
    """
    Construit / réordonne la matrice de features de manière chronologique.

    - HDFS :
        * on utilise les logs structurés pour calculer le premier timestamp par BlockId
        * on réordonne la matrice BlockId × EventId en conséquence
    - BGL :
        * la matrice est déjà agrégée par fenêtre temporelle (window_start)
        * on se contente éventuellement de trier sur window_start si présent
    """
    dataset = dataset.lower()

    # Charger la matrice existante
    df_matrix_chrono = pd.read_csv(matrix_csv)

    if dataset == "hdfs":
        print("[INFO] Chargement structured logs (HDFS)...")
        df_struct = load_structured_logs(structured_csv)

        print("[INFO] Calcul des timestamps de première apparition...")
        block_ts = compute_block_first_timestamp(df_struct)

        print("[INFO] Réordonnancement de la matrice EventId-block...")
        df_matrix_chrono = reorder_matrix_by_chronology(matrix_csv, block_ts)

    elif dataset == "bgl":
        print("[INFO] Dataset BGL : tri éventuel sur 'window_start' s'il est présent...")
        if "window_start" in df_matrix_chrono.columns:
            df_matrix_chrono["window_start"] = pd.to_datetime(
                df_matrix_chrono["window_start"], errors="coerce"
            )
            df_matrix_chrono = df_matrix_chrono.sort_values("window_start")
            print("[INFO] Tri chronologique appliqué sur 'window_start'.")
        else:
            print("[WARN] 'window_start' absent de la matrice BGL, aucun tri chronologique appliqué.")

    else:
        print(f"[WARN] Dataset '{dataset}' non reconnu, aucun réordonnancement spécifique appliqué.")

    print("[INFO] Sauvegarde du fichier final trié dans le temps...")
    df_matrix_chrono.to_csv(output_csv, index=False)
    print("[OK] Fichier produit :", output_csv)

