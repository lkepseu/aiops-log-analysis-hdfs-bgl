from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from configs.metrics_utils import compute_binary_classification_metrics, print_metrics


def train_eval_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Entraîne une régression logistique :
      - modèle initial : train (60%) -> eval sur val (20%)
      - modèle final   : train+val (80%) -> eval sur test (20%)
    """

    # --------- 1) Modèle initial (train → val) ---------
    logreg = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
    )
    logreg.fit(X_train, y_train)

    y_val_pred = logreg.predict(X_val)
    y_val_proba = logreg.predict_proba(X_val)[:, 1]

    metrics_val = compute_binary_classification_metrics(
        y_true=y_val,
        y_pred=y_val_pred,
        y_proba=y_val_proba,
    )
    print_metrics("Logistic Regression - Validation (60% → 20%)", metrics_val)

    # --------- 2) Modèle final (train+val → test) ---------
    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)

    logreg_final = LogisticRegression(
        solver="liblinear",
        max_iter=2000,
        class_weight="balanced",
    )
    logreg_final.fit(X_train_full, y_train_full)

    y_test_pred = logreg_final.predict(X_test)
    y_test_proba = logreg_final.predict_proba(X_test)[:, 1]

    metrics_test = compute_binary_classification_metrics(
        y_true=y_test,
        y_pred=y_test_pred,
        y_proba=y_test_proba,
    )
    print_metrics("Logistic Regression - Test (80% → 20%)", metrics_test)
