#!/usr/bin/env bash
set -euo pipefail

# Se placer à la racine du projet
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(pwd)"

#python3 $PROJECT_ROOT/3_model_contruction/reduce_multicollinearity.py \
#    --dataset hdfs \
#    --input $PROJECT_ROOT/data/features/HDFS_matrix_with_labels.csv \
#    --output $PROJECT_ROOT/data/features/HDFS_features_cleaned.csv \
#    --corr 0.7 \
#    --vif 10.0 \
#    --importances $PROJECT_ROOT/data/features/HDFS_feature_importances.csv

python3 $PROJECT_ROOT/3_model_contruction/reduce_multicollinearity.py \
    --dataset bgl \
    --input $PROJECT_ROOT/data/features/BGL_matrix_with_labels.csv \
    --output $PROJECT_ROOT/data/features/BLG_features_cleaned.csv \
    --corr 0.7 \
    --vif 10.0 \
#    --importances $PROJECT_ROOT/data/features/HDFS_feature_importances.csv
