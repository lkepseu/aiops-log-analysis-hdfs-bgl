import re
import pandas as pd
from configs.windows import apply_windows_by_session
from typing import Optional

# Étape 1. Fonction spécifique HDFS pour extraire le BlockId
BLOCK_REGEX = re.compile(r"blk_-?\d+")

def hdfs_block_id_extractor(row: pd.Series) -> Optional[str]:
    content = row.get("Content", "")
    match = BLOCK_REGEX.search(str(content))
    if match:
        return match.group(0)
    return None


# Étape 2. Fonction d'agrégation par BlockId → features
def hdfs_agg_event_id_histogram(df_block: pd.DataFrame, block_id: str) -> dict:
    counts = df_block["EventId"].value_counts().to_dict()
    return {
        "BlockId": block_id,
        **counts,
    }


# Étape 3. Fonction pour construire la matrice HDFS
def build_hdfs_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return apply_windows_by_session(
        df=df,
        session_extractor=hdfs_block_id_extractor,
        agg_func=hdfs_agg_event_id_histogram,
        session_col="BlockId",
    )
