import torch
from torch.utils.data import DataLoader
import pandas as pd
import tqdm
import os
import re


def make_predictions_neural_network(
    models: list,  # list of trained torch.nn.Module, one per fold
    dataloader: DataLoader,
    logger,
    predictions_save_dir: str = "/home/haris/results/predictions",
    project_name: str = "StockPredictor",
    device: str = "cuda",
    period_index: int = 0,
) -> pd.DataFrame:
    """
    Make predictions using multiple trained models (from K-fold CV) and log prediction errors.

    Returns the averaged predictions across folds as a pivot table (stock_code × date),
    and logs MAE, MSE, RMSE, IC, rank-IC for each period.

    Args:
        models (list of torch.nn.Module): Trained models for each fold.
        dataloader (DataLoader): DataLoader for prediction data (should yield date, stock_code, features, labels).
        logger (logging.Logger): Logger for process tracking.
        predictions_save_dir (str): Directory to save CSV files.
        device (str): Device to use ('cuda' or 'cpu').
        period_index (int): Optional index for logging different prediction periods.

    Returns:
        pd.DataFrame: Pivot table of averaged predictions (stock_code × date).
    """

    csv_path = os.path.join(predictions_save_dir, f"predictions_period_{period_index}.csv")
    if os.path.exists(csv_path):
        logger.info(f"Prediction file already exists, skip saving: {csv_path}")
        return None

    else:
        logger.info(f"Starting prediction with {len(models)} fold(s)...")

        # Move models to device and set eval mode
        for model in models:
            model.to(device)
            model.eval()

        all_fold_preds = []

        # Loop over folds
        for fold_idx, model in enumerate(models, start=1):
            fold_predictions, _, dates, stock_codes = [], [], [], []
            with torch.no_grad():
                for date, stock_code, features, labels in tqdm.tqdm(dataloader, desc=f"Predicting fold {fold_idx}", leave=False):
                    features = features.to(device)
                    outputs = model(features)
                    outputs = outputs.detach().cpu().numpy().squeeze()
                    date = date.detach().cpu().numpy().squeeze()
                    stock_code = stock_code.detach().cpu().numpy().squeeze()

                    fold_predictions.append(pd.Series(outputs))
                    dates.append(pd.Series(date))
                    stock_codes.append(pd.Series(stock_code))

            # Concatenate results for this fold
            fold_predictions = pd.concat(fold_predictions, ignore_index=True)
            dates = pd.concat(dates, ignore_index=True)
            stock_codes = pd.concat(stock_codes, ignore_index=True)
            stock_codes = stock_codes.apply(lambda x: str(int(x)).zfill(6))

            fold_df = pd.DataFrame({"date": dates, "stock_code": stock_codes, f"prediction_fold{fold_idx}": fold_predictions})
            all_fold_preds.append(fold_df)

        # Merge all folds
        result_df = all_fold_preds[0]
        for df in all_fold_preds[1:]:
            result_df = result_df.merge(df, on=["stock_code", "date"], how="outer")

        # Compute average prediction and label across folds
        pred_cols = [col for col in result_df.columns if col.startswith("prediction_fold")]
        result_df["prediction"] = result_df[pred_cols].mean(axis=1)

        # Pivot table: stock_code × date
        pivot_df = result_df.pivot(index="stock_code", columns="date", values="prediction")

        # 这里要读取最新的csv，然后覆盖掉对应的列，也就是每日实际上只增加了最后一列
        pattern = re.compile(r"predictions_period_(\d+)\.csv")
        existing_files = []
        for fname in os.listdir(predictions_save_dir):
            m = pattern.match(fname)
            if m:
                idx = int(m.group(1))
                if idx != period_index:
                    existing_files.append((idx, fname))

        if existing_files:
            # 找到 period_index 最大的历史文件
            latest_idx, latest_fname = max(existing_files, key=lambda x: x[0])
            latest_path = os.path.join(predictions_save_dir, latest_fname)
            history_df = pd.read_csv(latest_path, index_col=0)
            history_df.index = history_df.index.map(lambda x: str(x).zfill(6))
            new_date = pivot_df.columns[-1]
            history_df = history_df.reindex(history_df.index.union(pivot_df.index))
            history_df[new_date] = pivot_df[new_date]
            final_df = history_df
            logger.info(f"Loaded history from period {latest_idx}, updated column {new_date}")
        else:
            # 第一次预测，直接保存
            final_df = pivot_df
            logger.info("No historical prediction found, saving new file.")

        # 保存
        final_df.to_csv(csv_path)
        logger.info(f"Predictions saved to {csv_path}")

        return pivot_df


def make_all_period_predictions_neural_network(
    all_period_models: list,  # list of list of trained torch.nn.Module, one per period, each containing k_folds models
    dataloader: DataLoader,
    logger,
    predictions_save_dir: str = "/home/haris/results/predictions",
    project_name: str = "StockPredictor_20251231",
    device: str = "cuda",
    predict_date_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Make predictions using multiple trained models for different periods based on period index.

    For each data point, use the models from the corresponding period to make predictions.
    Returns the averaged predictions across folds as a pivot table (stock_code × date).

    Args:
        all_period_models (list): List of lists, where each inner list contains k_folds models for a specific period.
        dataloader (DataLoader): DataLoader for prediction data (should yield date, stock_code, features, labels).
        logger (logging.Logger): Logger for process tracking.
        predictions_save_dir (str): Directory to save CSV files.
        project_name (str): Project name for file naming.
        device (str): Device to use ('cuda' or 'cpu').
        predict_date_df (pd.DataFrame): DataFrame containing prediction dates and their corresponding period index.
    Returns:
        pd.DataFrame: Pivot table of averaged predictions (stock_code × date).
    """

    # Group data by period index
    period_indices = predict_date_df["period"].unique()
    latest_period = max(period_indices)

    logger.info(f"Found {len(period_indices)} periods: {sorted(period_indices)}")

    # Check if we have enough models for all periods
    if len(all_period_models) <= latest_period:
        raise ValueError(f"Not enough models provided. Need {latest_period} periods, but only have {len(all_period_models)} periods")

    # Initialize result storage for all folds across all periods
    all_fold_preds = []

    # Loop over folds
    for fold_idx in range(len(all_period_models[0])):
        fold_predictions, dates, stock_codes = [], [], []

        # Process dataloader for this fold
        for date, stock_code, features, labels in tqdm.tqdm(dataloader, desc=f"Predicting fold {fold_idx+1}", leave=False):
            # Get the appropriate model for this period and fold
            period_idx = predict_date_df.loc[str(date[0].detach().cpu().item()), "period"]
            period_models = all_period_models[period_idx]
            model = period_models[fold_idx]
            model.to(device)
            model.eval()
            with torch.no_grad():
                output = model(features.to(device))

            fold_predictions.append(pd.Series(output.detach().cpu().numpy().squeeze()))
            dates.append(pd.Series(date.detach().cpu().numpy().squeeze()))
            stock_codes.append(pd.Series(stock_code.detach().cpu().numpy().squeeze()))

        # Concatenate results for this fold
        fold_predictions = pd.concat(fold_predictions, ignore_index=True)
        dates = pd.concat(dates, ignore_index=True)
        stock_codes = pd.concat(stock_codes, ignore_index=True)
        stock_codes = stock_codes.apply(lambda x: str(x).zfill(6))

        fold_df = pd.DataFrame({"date": dates, "stock_code": stock_codes, f"prediction_fold{fold_idx+1}": fold_predictions})
        all_fold_preds.append(fold_df)

    # Merge all folds
    result_df = all_fold_preds[0]
    for df in all_fold_preds[1:]:
        result_df = result_df.merge(df, on=["stock_code", "date"], how="outer")

    # Compute average prediction across folds
    pred_cols = [col for col in result_df.columns if col.startswith("prediction_fold")]
    result_df["prediction"] = result_df[pred_cols].mean(axis=1)

    # Pivot table: stock_code × date
    pivot_df = result_df.pivot(index="stock_code", columns="date", values="prediction")

    # Save predictions
    csv_path = os.path.join(predictions_save_dir, f"{project_name}_history_all_stocks.csv")
    pivot_df.to_csv(csv_path)
    logger.info(f"Predictions saved to {csv_path}")

    return pivot_df
