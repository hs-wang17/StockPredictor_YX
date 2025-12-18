from typing import Tuple, Optional, List
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb
import time
import joblib


def train_lightgbm_model(
    model: lgb.LGBMRegressor,
    train_df: pd.DataFrame,
    logger,
    model_save_dir: str,
    period_index: int,
    project_name: str,
    k_folds: int = 4,
    early_stopping_rounds: int = 50,
    verbose_eval: int = 50,
    random_state: int = 42,
    use_gpu: bool = True,
    timestamp: str = None,
    feature_cols: list = None,
) -> List[lgb.LGBMRegressor]:
    """
    Train a LightGBM model.
    Parameters:
    - model: LightGBM model instance (lgb.LGBMRegressor).
    - train_df: DataFrame containing training data with features and target. The last column is assumed to be the target.
    - logger: Logger for logging information.
    - model_save_dir: Directory to save the trained model.
    - period_index: Index of the current training period (for logging and saving).
    - project_name: Name of the project (used in file naming).
    - k_folds: Number of folds for k-fold cross validation (default 4).
    - early_stopping_rounds: Number of rounds for early stopping.
    - verbose_eval: Frequency of logging during training.
    - random_state: Random seed for reproducibility.
    - use_gpu: Whether to use GPU for training if available.
    - timestamp: Timestamp string for file naming.
    - feature_cols: List of feature column names to use for training.
    Returns:
    - model: Trained LightGBM model (refitted on full data using average best iteration).
    """
    start_ts = time.time()
    model_save_dir = os.path.join(model_save_dir, f"{project_name}_{timestamp}", f"period_{period_index}_k{k_folds}")
    os.makedirs(model_save_dir, exist_ok=True)

    X, y = train_df.iloc[:, :-1][feature_cols], train_df["target"]

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    models = []
    fold_scores = []

    logger.info(f"Training {k_folds}-fold LightGBM ensemble")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_model = lgb.LGBMRegressor(**model.get_params())

        fit_kwargs = {}
        callbacks = []

        eval_set = [(X_val, y_val)]
        fit_kwargs["eval_set"] = eval_set
        fit_kwargs["eval_metric"] = "l2"

        if hasattr(lgb, "early_stopping"):
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))
        else:
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
        if hasattr(lgb, "log_evaluation"):
            callbacks.append(lgb.log_evaluation(period=verbose_eval))
        if callbacks:
            fit_kwargs["callbacks"] = callbacks

        fold_model.fit(X_train, y_train, **fit_kwargs)
        best_score = fold_model.best_score_["valid_0"]["l2"]
        fold_scores.append(best_score)

        model_path = os.path.join(model_save_dir, f"fold_{fold}.pkl")
        joblib.dump(fold_model, model_path)
        models.append(fold_model)

        logger.info(f"Fold {fold}/{k_folds} | " f"Val RMSE: {np.sqrt(best_score):.6f} | " f"Best iter: {getattr(fold_model, 'best_iteration_', 'N/A')}")

    logger.info(f"{k_folds}-fold CV RMSE: {np.sqrt(np.mean(fold_scores)):.6f} ± {np.sqrt(np.std(fold_scores)):.6f}")
    logger.info(f"All {k_folds} models saved in {model_save_dir} (total {time.time()-start_ts:.1f}s)")

    return models
