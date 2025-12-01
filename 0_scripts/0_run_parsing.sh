#!/usr/bin/env bash
set -euo pipefail

# Se placer AUTOMATIQUEMENT à la racine du projet
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Déterminer la racine du projet
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Parsing du dataset HDFS avec Drain
#python3 "$PROJECT_ROOT/1_logparser/parse_with_drain.py" --dataset HDFS

# Parsing du dataset BGL avec Drain
python3 "$PROJECT_ROOT/1_logparser/parse_with_drain.py" --dataset BGL
