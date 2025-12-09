#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(pwd)"

python3 "$PROJECT_ROOT/2_features_extraction/generate_features_matrix.py" --dataset bgl \
 --input "$PROJECT_ROOT/data/parsed/BGL/BGL.log_structured.csv" \
 --output "$PROJECT_ROOT/data/features/BGL_matrix.csv" \
 --timestamp-col Time --window-minutes 5 --step-minutes 5

python3 "$PROJECT_ROOT/2_features_extraction/generate_features_matrix.py" --dataset hdfs \
 --input "$PROJECT_ROOT/data/parsed/HDFS/HDFS.log_structured.csv" \
 --output "$PROJECT_ROOT/data/features/HDFS_matrix.csv"
