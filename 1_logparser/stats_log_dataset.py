#!/usr/bin/env python3

import argparse
import pandas as pd
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s"
    )

def compute_stats(structured_csv_path: str, dataset: str):
    logging.info(f"Chargement du CSV : {structured_csv_path}")
    df = pd.read_csv(structured_csv_path)

    logging.info("=== Statistiques générales ===")
    print(df.describe(include="all"))

    logging.info("=== Occurrences par EventId ===")
    print(df["EventId"].value_counts())

    if dataset == "HDFS":
        if "Content" in df.columns:
            df["BlockId"] = df["Content"].str.extract(r"(blk_[\-]?\d+)")
            logging.info("=== Statistiques par BlockId ===")
            print(df.groupby("BlockId")["EventId"].count().describe())
        else:
            logging.warning("Pas de colonne 'Content' pour extraire les BlockId.")

    if dataset == "BGL":
        node_col = "Node" if "Node" in df.columns else "Location"
        if node_col in df.columns:
            logging.info("=== Statistiques par Node ===")
            print(df.groupby(node_col)["EventId"].count().describe())
        else:
            logging.warning("Pas de colonne 'Node' ou 'Location' pour BGL.")

def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Affiche des statistiques avancées sur un dataset HDFS ou BGL."
    )

    parser.add_argument("--dataset", type=str, required=True, choices=["HDFS","BGL"])
    parser.add_argument("--structured", required=True, help="CSV structuré Drain")

    args = parser.parse_args()
    compute_stats(args.structured, args.dataset)

if __name__ == "__main__":
    main()
