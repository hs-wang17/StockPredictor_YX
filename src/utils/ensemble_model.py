from typing import Any, Dict, Optional
import lightgbm as lgb
import joblib
import os

def create_lgbm_model(params: Optional[Dict[str, Any]] = None, n_estimators: int = 1000, random_state: int = 42) -> lgb.LGBMRegressor:
    """
    创建 LGBM 回归器 (sklearn API wrapper)。
    params: lightgbm 参数字典（会覆盖默认参数）。
    """
    default_params = {
        "objective": "regression",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "random_state": random_state,
        "device_type": 'cuda',
    }
    if params:
        default_params.update(params)
    model = lgb.LGBMRegressor(n_estimators=n_estimators, **default_params)
    return model

def save_lgbm_model(model: lgb.LGBMRegressor, file_path: str):
    """
    保存整个 sklearn-wrapped LGBM 模型（使用 joblib，方便 reload 继续 predict）。
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)

def load_lgbm_model(file_path: str) -> lgb.LGBMRegressor:
    """
    从文件加载模型。
    """
    model = joblib.load(file_path)
    return model
