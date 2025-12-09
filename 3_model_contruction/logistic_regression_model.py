from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

from configs.metrics_utils import compute_binary_classification_metrics, print_metrics


def train_eval_logistic_regression_timeseries(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> None:
    """
    Entraîne et évalue une régression logistique avec une stratégie de
    TimeSeries Cross-Validation (Rolling Window / Sliding-origin).

    - Les splits respectent strictement l'ordre temporel (X et y doivent être triés).
    - À chaque fold, le modèle est entraîné sur le passé et évalué sur un segment futur.
    - Aucune fuite du futur dans le passé.
    - Les données de test NE SONT PAS rebalancées : on conserve la vraie distribution
      (faible taux d'anomalies).
    - Le rééquilibrage est géré uniquement via class_weight="balanced" sur le train.

    Paramètres
    ----------
    X : pd.DataFrame
        Matrice de features triée chronologiquement (index croissant = temps).
    y : pd.Series
        Labels binaires correspondants (0 = normal, 1 = anomalie).
    n_splits : int
        Nombre de folds pour la TimeSeries Cross-Validation.
    """

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # ----- Modèle entraîné uniquement sur le passé -----
        logreg = LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            class_weight="balanced",  # pondération seulement sur le TRAIN
        )
        logreg.fit(X_train, y_train)

        # ----- Évaluation sur le futur (test du fold) -----
        # IMPORTANT : test jamais rebalancé → distribution réelle conservée
        y_test_pred = logreg.predict(X_test)
        y_test_proba = logreg.predict_proba(X_test)[:, 1]

        metrics = compute_binary_classification_metrics(
            y_true=y_test,
            y_pred=y_test_pred,
            y_proba=y_test_proba,
        )

        title = f"Logistic Regression - TimeSeriesCV Fold {fold_idx}"
        print_metrics(title, metrics)
        fold_metrics.append((title, metrics))

    # Optionnel : dernier fold considéré comme "test final"
    last_title, last_metrics = fold_metrics[-1]
    print("\n=== Dernier fold considéré comme Test final (futur) ===")
    print_metrics("Logistic Regression - Test final (dernier fold)", last_metrics)
