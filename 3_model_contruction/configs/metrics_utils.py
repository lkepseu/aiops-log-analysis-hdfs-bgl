from typing import Dict, Any, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


def compute_binary_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Calcule un ensemble de métriques de classification binaire.
    y_true : labels vrais (0/1)
    y_pred : prédictions binaires (0/1)
    y_proba : probabilités associées à la classe positive (optionnel)
    """
    metrics: Dict[str, Any] = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = np.nan

        try:
            metrics["pr_auc"] = average_precision_score(y_true, y_proba)
        except ValueError:
            metrics["pr_auc"] = np.nan

    return metrics


def print_metrics(title: str, metrics: Dict[str, Any]) -> None:
    """
    Affiche proprement les métriques.
    """
    print(f"\n=== {title} ===")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name:>9}: {value:.4f}")
        else:
            print(f"{name:>9}: {value}")
