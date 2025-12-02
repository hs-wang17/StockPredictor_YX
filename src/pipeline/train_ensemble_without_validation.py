from typing import Tuple, Optional
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
    early_stopping_rounds: int = 50,
    valid_size: float = 0.1,
    verbose_eval: int = 50,
    random_state: int = 42,
    use_gpu: bool = True,
    timestamp: str = None,
    feature_cols: list = None,
) -> lgb.LGBMRegressor:
    """
    Train a LightGBM model.
    Parameters:
    - model: LightGBM model instance (lgb.LGBMRegressor).
    - train_df: DataFrame containing training data with features and target. The last column is assumed to be the target.
    - logger: Logger for logging information.
    - model_save_dir: Directory to save the trained model.
    - period_index: Index of the current training period (for logging and saving).
    - project_name: Name of the project (used in file naming).
    - early_stopping_rounds: Number of rounds for early stopping.
    - valid_size: Proportion of data to use for validation (between 0 and 1). If 0, no validation is done.
    - verbose_eval: Frequency of logging during training.
    - random_state: Random seed for reproducibility.
    - use_gpu: Whether to use GPU for training if available.
    - timestamp: Timestamp string for file naming.
    - feature_cols: List of feature column names to use for training.
    Returns:
    - model: Trained LightGBM model.
    """
    start_ts = time.time()
    model_save_dir = os.path.join(model_save_dir, f"{project_name}_{timestamp}")
    os.makedirs(model_save_dir, exist_ok=True)

    X, y = train_df.iloc[:, :-1], train_df["target"]
    if valid_size > 0 and len(X) > 0:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=valid_size, random_state=random_state, shuffle=True)
        eval_set = [(X_val[feature_cols], y_val)]
    else:
        X_train, y_train = X, y
        eval_set = None

    logger.info(f"Training LightGBM: n_train={len(X_train)}, n_val={len(X_val) if valid_size > 0 else 0}")

    # fit with early stopping if validation set provided
    fit_kwargs = {}
    callbacks = []

    if eval_set is not None:
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

    model.fit(X_train[feature_cols], y_train, **fit_kwargs)

    model_file = os.path.join(model_save_dir, f"{project_name}_{timestamp}_period_{period_index}_lgbm.pkl")
    joblib.dump(model, model_file)
    logger.info(f"Model saved to {model_file} (fit time {time.time()-start_ts:.1f}s)")

    return model
