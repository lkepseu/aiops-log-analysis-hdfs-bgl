#!/usr/bin/env bash
set -euo pipefail

# Se placer AUTOMATIQUEMENT à la racine du projet
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Déterminer la racine du projet
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Inspection des fichiers templates générés pour HDFS
python3 "$PROJECT_ROOT/1_logparser/inspect_parsed_data.py" \
    --parsed-dir "$PROJECT_ROOT/data/parsed/HDFS"

# Inspection des fichiers templates générés pour BGL
python3 "$PROJECT_ROOT/1_logparser/inspect_parsed_data.py" \
    --parsed-dir "$PROJECT_ROOT/data/parsed/BGL"
