from typing import Tuple, Optional
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import time
import joblib
from datetime import datetime

def train_lightgbm_model(
    model: lgb.LGBMRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    logger,
    model_save_dir: str,
    period_index: int,
    project_name: str,
    early_stopping_rounds: int = 50,
    valid_size: float = 0.1,
    verbose_eval: int = 50,
    random_state: int = 42,
    use_gpu: bool = True
) -> lgb.LGBMRegressor:
    """
    训练 LightGBM 模型并保存。
    - X, y: 整个训练集（DataFrame / Series），会随机划出一小部分做验证（若 valid_size=0 则不做验证）。
    - 返回训练好的模型（sklearn API wrapper）。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_ts = time.time()
    os.makedirs(model_save_dir, exist_ok=True)
    device_type = "gpu" if use_gpu else "cpu"

    if valid_size > 0 and len(X) > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=valid_size, random_state=random_state, shuffle=True
        )
        eval_set = [(X_val, y_val)]
    else:
        X_train, y_train = X, y
        eval_set = None

    logger.info(f"Training LightGBM: n_train={len(X_train)}, n_val={len(X_val) if valid_size>0 else 0}")

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

    model.fit(X_train, y_train, **fit_kwargs)

    model_file = os.path.join(model_save_dir, f"{project_name}_{timestamp}_period{period_index}_lgbm.pkl")
    joblib.dump(model, model_file)
    logger.info(f"Model saved to {model_file} (fit time {time.time()-start_ts:.1f}s)")

    return model
