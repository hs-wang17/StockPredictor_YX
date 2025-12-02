from typing import Tuple
import os
import pandas as pd
import numpy as np


def make_predictions_lightgbm(
    model,
    predict_df: pd.DataFrame,
    logger,
    model_save_dir: str = "/home/user0/results/models",
    predictions_save_dir: str = "/home/user0/results/predictions",
    project_name: str = "StockPredictor",
    period_index: int = 0,
    date_col: str = "date",
    code_col: str = "code",
    timestamp: str = None,
    feature_cols: list = None,
) -> pd.DataFrame:
    """
    使用训练好的 model 对 predict_df（单个 period 的合并 DataFrame）进行预测。
    predict_df 必须包含 date_col, code_col, feature_cols。
    返回包含 ['date','code','prediction'] 的 DataFrame（并保存 CSV）。
    """
    predictions_save_dir = os.path.join(predictions_save_dir, f"{project_name}_{timestamp}")
    os.makedirs(predictions_save_dir, exist_ok=True)
    X_pred = predict_df.iloc[:, :-1]
    preds = model.predict(X_pred[feature_cols], num_iteration=model.best_iteration_) if hasattr(model, "best_iteration_") else model.predict(X_pred)

    out_df = pd.DataFrame({"date": predict_df[date_col].values, "stock_code": predict_df[code_col].values, "prediction": preds}).pivot(
        index="stock_code", columns="date", values="prediction"
    )
    csv_path = os.path.join(predictions_save_dir, f"predictions_period_{period_index}.csv")
    out_df.to_csv(csv_path)
    logger.info(f"Predictions saved to {csv_path}")

    return out_df
