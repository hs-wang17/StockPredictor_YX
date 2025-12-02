from typing import List
import os
import pandas as pd
import numpy as np


def make_predictions_lightgbm(
    models: List,  # List of trained LightGBM models (e.g., from k-fold training)
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
    Perform ensemble prediction by averaging predictions from a list of trained LightGBM models.

    Args:
        models: List of trained lgb.LGBMRegressor (typically k-fold models)
        predict_df: DataFrame with features, date_col, and code_col
        logger: Logger instance
        model_save_dir: Directory to save trained models
        predictions_save_dir: Directory to save prediction CSV
        project_name: Project name (for output path)
        period_index: Current period index
        date_col: Name of the date column
        code_col: Name of the stock code column
        timestamp: Timestamp string (for output path)
        feature_cols: List of feature column names

    Returns:
        Pivoted DataFrame: stock_code (rows) * date (columns) with ensemble predictions
    """
    predictions_save_dir = os.path.join(predictions_save_dir, f"{project_name}_{timestamp}")
    os.makedirs(predictions_save_dir, exist_ok=True)

    X_pred = predict_df.iloc[:, :-1][feature_cols]
    logger.info(f"Ensemble predicting with {len(models)} models")

    fold_preds = []
    for model in models:
        pred = model.predict(X_pred, num_iteration=getattr(model, "best_iteration_", None) or None)
        fold_preds.append(pred)

    final_preds = np.mean(fold_preds, axis=0)

    out_df = pd.DataFrame({"date": predict_df[date_col].values, "stock_code": predict_df[code_col].values, "prediction": final_preds}).pivot(
        index="stock_code", columns="date", values="prediction"
    )

    csv_path = os.path.join(predictions_save_dir, f"predictions_period_{period_index}.csv")
    out_df.to_csv(csv_path)
    logger.info(f"Predictions saved to {csv_path}")

    return out_df
