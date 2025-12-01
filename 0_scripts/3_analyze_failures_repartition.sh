#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(pwd)"

python3 $PROJECT_ROOT/2_features_extraction/analyze_failures_repartition.py \
    --dataset hdfs \
    --matrix_csv $PROJECT_ROOT/data/features/HDFS_matrix.csv \
    --labels_csv $PROJECT_ROOT/data/raw/HDFS_anomaly_label.csv \
    --top_k 20

#python3 $PROJECT_ROOT/2_features_extraction/analyze_failures_repartition.py \
#    --dataset bgl \
#    --matrix_csv $PROJECT_ROOT/data/features/BGL_matrix.csv \
#    --structured_csv $PROJECT_ROOT/data/parsed/BGL/BGL.log_structured.csv
