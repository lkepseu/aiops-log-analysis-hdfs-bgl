# inspect_parsed_data.py
import pandas as pd
from pathlib import Path
import argparse


# Afficher les 10 premières lignes des fichiers *_templates.csv
def show_templates(parsed_dir: str):
    # Etape 1. Récupérer le chemin vers le dossier contenant les fichiers générés par Drain
    parsed_path = Path(parsed_dir)

    # Étape 2. Parcourir tous les fichiers templates
    for csv in parsed_path.glob("*templates.csv"):
        print(f"=== {csv.name} ===")
        # Charger le CSV dans un DataFrame
        df = pd.read_csv(csv)
        # Afficher un aperçu (head = 10 lignes)
        print(df.head(10))
        print()

# Construction du parser d’arguments CLI
def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspection des fichiers templates Drain (EventId, EventTemplate, Occurrences)."
    )

    parser.add_argument(
        "--parsed-dir",
        type=str,
        required=True,
        help="Chemin vers le dossier contenant les fichiers structurés/templates (ex: data/parsed/HDFS).",
    )

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    show_templates(args.parsed_dir)
