#!/usr/bin/env bash
set -euo pipefail

# Se placer à la racine du projet
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(pwd)"

echo "[RUN] Entraînement modèles HDFS (TimeSeries CV)"
python3 $PROJECT_ROOT/3_model_contruction/train_models.py \
  --dataset hdfs \
  --matrix_csv $PROJECT_ROOT/data/features/HDFS_features_cleaned.csv \
  --labels_csv $PROJECT_ROOT/data/raw/HDFS_anomaly_label.csv \
  --splits 5 \
  --model both

echo "[RUN] Entraînement modèles BGL (TimeSeries CV)"
python3 $PROJECT_ROOT/3_model_contruction/train_models.py \
  --dataset bgl \
  --matrix_csv $PROJECT_ROOT/data/features/BLG_features_cleaned.csv \
  --structured_csv $PROJECT_ROOT/data/parsed/BGL/BGL.log_structured.csv \
  --splits 5 \
  --model both
