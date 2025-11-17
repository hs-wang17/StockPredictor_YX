import torch
from torch.utils.data import DataLoader
import pandas as pd
import tqdm
import numpy as np
import os
import swanlab


def make_predictions_neural_network(
    models: list,  # list of trained torch.nn.Module, one per fold
    dataloader: DataLoader,
    logger,
    predictions_save_dir: str = "/home/user0/results/predictions",
    device: str = "cuda",
    use_swanlab: bool = True,
    period_index: int = 0,
    model_save_dir: str = "/home/user0/results/models/",
    timestamp: str = None,
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
    os.makedirs(predictions_save_dir, exist_ok=True)
    logger.info(f"Starting prediction with {len(models)} fold(s)...")

    # Move models to device and set eval mode
    for model in models:
        model.to(device)
        model.eval()

    all_fold_preds = []

    # Loop over folds
    for fold_idx, model in enumerate(models, start=1):
        fold_predictions, fold_labels, dates, stock_codes = [], [], [], []
        with torch.no_grad():
            for date, stock_code, features, labels in tqdm.tqdm(dataloader, desc=f"Predicting fold {fold_idx}", leave=False):
                features = features.to(device)
                outputs = model(features)
                outputs = outputs.detach().cpu().numpy().squeeze()
                labels = labels.detach().cpu().numpy().squeeze()
                date = date.detach().cpu().numpy().squeeze()
                stock_code = stock_code.detach().cpu().numpy().squeeze()

                fold_predictions.append(pd.Series(outputs))
                fold_labels.append(pd.Series(labels))
                dates.append(pd.Series(date))
                stock_codes.append(pd.Series(stock_code))

        # Concatenate results for this fold
        fold_predictions = pd.concat(fold_predictions, ignore_index=True)
        fold_labels = pd.concat(fold_labels, ignore_index=True)
        dates = pd.concat(dates, ignore_index=True)
        stock_codes = pd.concat(stock_codes, ignore_index=True)
        stock_codes = stock_codes.apply(lambda x: str(int(x)).zfill(6))

        fold_df = pd.DataFrame({"date": dates, "stock_code": stock_codes, f"prediction_fold{fold_idx}": fold_predictions, f"label_fold{fold_idx}": fold_labels})
        all_fold_preds.append(fold_df)

    # Merge all folds
    result_df = all_fold_preds[0]
    for df in all_fold_preds[1:]:
        result_df = result_df.merge(df, on=["stock_code", "date"], how="outer")

    # Compute average prediction and label across folds
    pred_cols = [col for col in result_df.columns if col.startswith("prediction_fold")]
    label_cols = [col for col in result_df.columns if col.startswith("label_fold")]

    result_df["prediction"] = result_df[pred_cols].mean(axis=1)
    result_df["label"] = result_df[label_cols].mean(axis=1)

    # Compute error metrics
    abs_error = (result_df["prediction"] - result_df["label"]).abs()
    sq_error = (result_df["prediction"] - result_df["label"]) ** 2
    mae = abs_error.mean()
    mse = sq_error.mean()
    rmse = np.sqrt(mse)
    ic = np.corrcoef(result_df["prediction"], result_df["label"])[0, 1] if len(result_df) > 1 else np.nan
    rank_ic = result_df["prediction"].rank().corr(result_df["label"].rank()) if len(result_df) > 1 else np.nan

    logger.info(f"Prediction stats for period {period_index}: MAE={mae:.6f}, RMSE={rmse:.6f}, IC={ic:.4f}, rank-IC={rank_ic:.4f}")

    # SwanLab logging
    if use_swanlab:
        swanlab.log(
            {
                f"pred_mean": result_df["prediction"].mean(),
                f"pred_std": result_df["prediction"].std(),
                f"pred_metric_MAE": mae,
                f"pred_metric_MSE": mse,
                f"pred_metric_RMSE": rmse,
                f"pred_metric_IC": ic,
                f"pred_metric_rank_IC": rank_ic,
            }
        )

    # Pivot table: stock_code × date
    pivot_df = result_df.pivot(index="stock_code", columns="date", values="prediction")

    # Save predictions to CSV
    csv_path = os.path.join(predictions_save_dir, f"predictions_period_{period_index}.csv")
    pivot_df.to_csv(csv_path)
    logger.info(f"Predictions saved to {csv_path}")

    return pivot_df
