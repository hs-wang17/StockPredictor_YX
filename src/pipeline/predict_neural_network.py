import torch
from torch.utils.data import DataLoader
import pandas as pd
import tqdm
import numpy as np
import os
import swanlab

def make_predictions_neural_network(
        model: torch.nn.Module,
        dataloader: DataLoader,
        logger,
        predictions_save_dir: str = '/home/user0/results/predictions',
        device: str = 'cuda',
        use_swanlab: bool = True
    ) -> pd.DataFrame:
    """
    Make predictions using the trained model and provided DataLoader.
    Save predictions with corresponding stock codes into a CSV file.
    
    Args:
        model (torch.nn.Module): The trained model.
        dataloader (DataLoader): DataLoader containing the data to make predictions on.
        logger (logging.Logger): Logger for process tracking.
        predictions_save_dir (str): Directory to save the prediction CSV files.
        device (str): Device to use ('cuda' or 'cpu').
    
    Returns:
        pd.DataFrame: The predictions as a pivot table (stock_code × date).
    """
    os.makedirs(predictions_save_dir, exist_ok=True)
    model = model.to(device)
    model.eval()

    predictions, dates, stock_codes = [], [], []
    logger.info("Starting prediction...")

    with torch.no_grad():
        for date, stock_code, features, labels in tqdm.tqdm(dataloader, desc="Predicting", leave=False):
            # Move tensors to device
            features = features.to(device)
            
            # Forward pass
            outputs = model(features)

            # Move results back to CPU for processing
            outputs = outputs.detach().cpu().numpy().squeeze()
            date = date.detach().cpu().numpy().squeeze()
            stock_code = stock_code.detach().cpu().numpy().squeeze()

            # Collect results
            predictions.append(pd.Series(outputs))
            dates.append(pd.Series(date))
            stock_codes.append(pd.Series(stock_code))

    # Concatenate predictions
    predictions = pd.concat(predictions, ignore_index=True)
    dates = pd.concat(dates, ignore_index=True)
    stock_codes = pd.concat(stock_codes, ignore_index=True)
    stock_codes = stock_codes.apply(lambda x: str(int(x)).zfill(6))

    # Assemble final DataFrame
    result_df = pd.DataFrame({
        'date': dates,
        'stock_code': stock_codes,
        'prediction': predictions
    }).pivot(index='stock_code', columns='date', values='prediction')

    logger.info("Prediction completed successfully.")

    if use_swanlab:
        swanlab.log({
            "prediction/mean": np.mean(predictions),
            "prediction/std": np.std(predictions),
        })

    return result_df
