# parse_with_drain.py
#
# Script de parsing de logs avec Drain (projet logpai/logparser).
# ---------------------------------------------------------------
# Ce fichier constitue le point d'entrée du pipeline AIOps :
#   - Chargement de la configuration adaptée (HDFS ou BGL)
#   - Application des regex (pré-normalisation)
#   - Parsing avec l’algorithme Drain (construction de l’arbre)
#   - Génération des fichiers CSV structurés et des templates
#
#

import argparse
import logging
import os
from pathlib import Path
from configs.remap_event_ids import remap_event_ids
from configs.parsing_config import get_parsing_configs

try:
    # Cas des installations classiques via pip
    from logparser.Drain import LogParser
except ImportError:
    # Cas où le module est installé sous un namespace différent
    from logparser.drain import LogParser

# Configurer un système de logging simple pour le suivi les tests en ligne de commande.
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )

# S'assurer que le dossier d'export ou d'import des données existe.
def ensure_directory(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


# Parser un dataset donné (HDFS ou BGL) en utilisant Drain.
def parse_dataset(dataset_name: str):

    # Etape 1. Récuperer la configuration adaptée au dataset.
    configs = get_parsing_configs()
    if dataset_name not in configs:
        raise ValueError(
            f"Dataset inconnu: {dataset_name}. "
            f"Datasets disponibles: {list(configs.keys())}"
        )
    cfg = configs[dataset_name]

    logging.info(f"=== Parsing du dataset {cfg.dataset_name} avec Drain ===")

    # Etape 2. Préparer du dossier de sortie
    ensure_directory(cfg.outdir)

    # Etape 3. Instancier le parser Drain
    parser = LogParser(
        log_format=cfg.log_format,
        indir=cfg.indir,
        outdir=cfg.outdir,
        depth=cfg.depth,
        st=cfg.st,
        rex=cfg.rex
    )

    # Etape 4. Parsing effectif du fichier
    parser.parse(cfg.log_file)

    # Fichiers produits par logpai/logparser
    structured_path = os.path.join(cfg.outdir, f"{cfg.log_file}_structured.csv")
    templates_path = os.path.join(cfg.outdir, f"{cfg.log_file}_templates.csv")

    # Étape 5. Remapping des EventId hexadecimal vers E1, E2, E3, ...
    remap_event_ids(templates_path=templates_path, structured_path=structured_path)

    logging.info(f"Fichier structuré  : {structured_path}")
    logging.info(f"Fichier templates  : {templates_path}")
    logging.info("=== Parsing terminé ===")


def main():
    # Demarrage du système de logs
    setup_logging()

    # Utilisation de argparse pour simplifier les appels en CLI.
    parser = argparse.ArgumentParser(
        description="Parsing de logs HDFS/BGL avec Drain (logpai/logparser)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["HDFS", "BGL"],
        help="Nom du dataset à parser (HDFS ou BGL).",
    )
    args = parser.parse_args()


    # Parsing des données de logs avec Drain
    parse_dataset(args.dataset)


if __name__ == "__main__":
    main()
