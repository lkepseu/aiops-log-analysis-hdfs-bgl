import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

from configs.metrics_utils import compute_binary_classification_metrics, print_metrics


def train_eval_random_forest_timeseries(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> None:
    """
    Entraîne et évalue un RandomForest avec une stratégie de TimeSeries Cross-Validation
    (Rolling Window / Sliding-origin evaluation) :

      - Les indices de train/test respectent strictement l'ordre temporel.
      - À chaque fold, la frontière train/test est déplacée vers le futur.
      - Aucune fuite du futur vers le passé.
      - Les données de test NE SONT PAS rebalancées (distribution réelle conservée).
      - Le modèle est pondéré via class_weight="balanced" uniquement sur le train.

    Paramètres
    ----------
    X : pd.DataFrame
        Matrice de features triée chronologiquement (index croissant = temps).
    y : pd.Series
        Labels binaires correspondants (0 = normal, 1 = anomalie).
    n_splits : int
        Nombre de folds pour la TimeSeries Cross-Validation.
    """

    # TimeSeriesSplit = Rolling Window (train = passé, test = futur)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # ---- Modèle entraîné uniquement sur le passé ----
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced",  # repondération uniquement sur le TRAIN
            random_state=42,
        )
        rf.fit(X_train, y_train)

        # ---- Évaluation sur le futur (test du fold) ----
        # IMPORTANT : test jamais rebalancé -> distribution réelle conservée
        y_test_pred = rf.predict(X_test)
        y_test_proba = rf.predict_proba(X_test)[:, 1]

        metrics = compute_binary_classification_metrics(
            y_true=y_test,
            y_pred=y_test_pred,
            y_proba=y_test_proba,
        )

        title = f"Random Forest - TimeSeriesCV Fold {fold_idx}"
        print_metrics(title, metrics)
        fold_metrics.append((title, metrics))

    # Optionnel : on peut désigner le DERNIER fold comme "Test final (futur)"
    last_title, last_metrics = fold_metrics[-1]
    print("\n=== Dernier fold considéré comme Test final (futur) ===")
    print_metrics("Random Forest - Test final (dernier fold)", last_metrics)
