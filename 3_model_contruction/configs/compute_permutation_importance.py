import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def compute_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Calcule la permutation importance d'un modèle scikit-learn
    en utilisant une TimeSeries Cross-Validation (Rolling / Sliding-origin).

    - À chaque fold :
        * le modèle est ré-entraîné sur le passé (train)
        * la permutation importance est calculée sur le futur (test)
    - Les données de TEST restent intactes (pas de rééchantillonnage, pas de balancing).
    - Le modèle est ré-entraîné SANS class_weight pour l’interprétation.

    Retourne : DataFrame triée par importance décroissante,
               importance moyenne et écart-type sur les n_splits.
    """

    # TimeSeries CV (respect du temps)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    n_features = X.shape[1]
    all_importances_mean = []
    all_importances_std = []

    print(
        f"[INFO] Permutation Importance avec TimeSeriesSplit "
        f"(n_splits={n_splits}, n_repeats={n_repeats})"
    )

    # On récupère les hyperparamètres du modèle, sans class_weight
    base_params = model.get_params()
    if "class_weight" in base_params:
        base_params.pop("class_weight")

    fold_idx = 0
    for train_idx, test_idx in tscv.split(X):
        fold_idx += 1
        print(f"\n[INFO] Fold {fold_idx} — Train: {len(train_idx)} | Test: {len(test_idx)}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # --- Ré-entraînement du modèle pour l’interprétation (sans class_weight) ---
        print("[INFO]   Ré-entraînement du modèle (sans class_weight) sur le train du fold...")
        model_clean = model.__class__(**base_params)
        model_clean.fit(X_train, y_train)

        # --- Calcul de la permutation importance sur le test du fold ---
        print("[INFO]   Calcul de la permutation importance sur le test du fold...")
        perm = permutation_importance(
            model_clean,
            X_test,
            y_test,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring="average_precision",  # PR-AUC : adapté aux datasets déséquilibrés
            n_jobs=-1,
        )

        all_importances_mean.append(perm.importances_mean)
        all_importances_std.append(perm.importances_std)

    # --- Agrégation sur les folds ---
    all_importances_mean = np.vstack(all_importances_mean)  # (n_splits, n_features)
    all_importances_std = np.vstack(all_importances_std)    # (n_splits, n_features)

    mean_importance = all_importances_mean.mean(axis=0)
    std_importance = all_importances_mean.std(axis=0)  # écart-type des moyennes par fold

    df_importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": mean_importance,
            "importance_std": std_importance,
        }
    ).sort_values("importance_mean", ascending=False)

    print("\n[RESULT] Top 15 features (Permutation Importance, moyennée sur les folds) :")
    print(df_importance.head(15))

    try:


            top_k = 15
            df_top = df_importance.head(top_k).iloc[::-1]  # reverse pour barh

            plt.figure(figsize=(10, 7))
            plt.barh(df_top["feature"], df_top["importance_mean"], xerr=df_top["importance_std"], alpha=0.8)
            plt.xlabel("Permutation Importance (mean decrease in PR-AUC)")
            plt.ylabel("Feature")
            plt.title(f"Top {top_k} Features – Permutation Importance (moyenne sur {n_splits} folds)")
            plt.tight_layout()

            output_path = f"importance_top_{top_k}_{model.__class__.__name__.lower()}.png"
            plt.savefig(output_path, dpi=300)
            plt.close()

            print(f"[PLOT] Bar plot sauvegardé dans : {output_path}")

    except Exception as e:
        print(f"[WARN] Impossible de générer le bar plot : {e}")

    return df_importance
