from typing import Tuple
import os
import pandas as pd
import numpy as np

def make_predictions_lightgbm(
    model,
    predict_df: pd.DataFrame,
    feature_cols,
    logger,
    predictions_save_dir: str,
    project_name: str,
    period_index: int,
    date_col: str = "date",
    code_col: str = "code"
) -> pd.DataFrame:
    """
    使用训练好的 model 对 predict_df（单个 period 的合并 DataFrame）进行预测。
    predict_df 必须包含 date_col, code_col, feature_cols。
    返回包含 ['date','code','prediction'] 的 DataFrame（并保存 CSV）。
    """
    os.makedirs(predictions_save_dir, exist_ok=True)
    X_pred = predict_df[feature_cols]
    preds = model.predict(X_pred, num_iteration=model.best_iteration_) if hasattr(model, "best_iteration_") else model.predict(X_pred)

    out_df = pd.DataFrame({
        "date": predict_df[date_col].values,
        "stock_code": predict_df[code_col].values,
        "prediction": preds
    }).pivot(index='stock_code', columns='date', values='prediction')

    return out_df
