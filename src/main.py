import pipeline.data as pipeline_data
import pipeline.filter as pipeline_filter
import pipeline.predict as pipeline_predict
import pipeline.train as pipeline_train
import utils.function as utils_function
import utils.logger as utils_logger
import utils.model as utils_model

import numpy as np
import pandas as pd
import os
import argparse
import tqdm

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the stock prediction pipeline.")
    
    parser.add_argument(
        '--train_period_years', type=int, default=3, help="Number of years for the training period (default: 3)"
    )
    parser.add_argument(
        '--predict_period_months', type=int, default=3, help="Number of months for the prediction period (default: 3)"
    )
    parser.add_argument(
        '--slide_period_months', type=int, default=3, help="Number of months for the sliding window (default: 3)"
    )
    parser.add_argument(
        '--data_dir', type=str, default='/home/user0/data/StockDailyData/', help="Directory containing stock data files"
    )
    parser.add_argument(
        '--num_periods', type=int, default=None, help="Number of periods to process (default: all)"
    )
    parser.add_argument(
        '--log_dir', type=str, default='/home/user0/results/logs', help="Directory to save logs"
    )
    parser.add_argument(
        '--model_dir', type=str, default='/home/user0/results/models', help="Directory to save models"
    )
    parser.add_argument(
        '--results_dir', type=str, default='/home/user0/results/predictions', help="Directory to save predictions"
    )
    parser.add_argument(
        '--batch_size', type=int, default=64, help="Batch size for model training (default: 64)"
    )
    parser.add_argument(
        '--filter_file_path', type=str, default='config/filter_index.fea', help="Path to filter index file"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()

    # Setup logger
    logger = utils_logger.setup_logger(log_dir=args.log_dir)

    # Generate training and prediction periods
    date_list = sorted([file_name[:8] for file_name in os.listdir(args.data_dir)])
    num_periods, train_dates_list, predict_dates_list = utils_function.generate_train_predict_dates(
        date_list, 
        train_period_years=args.train_period_years, 
        predict_period_months=args.predict_period_months, 
        slide_period_months=args.slide_period_months
    )

    logger.info(f"Number of periods: {num_periods}")
    logger.info("=" * 40)

    for i in range(num_periods)[:2]:
        logger.info(f"Period {i+1}:")
        logger.info(f"  Train Dates: {train_dates_list[i]}")
        logger.info(f"  Predict Dates: {predict_dates_list[i]}")
        # Load and preprocess data for the current period
        train_date_list, predict_date_list = train_dates_list[i], predict_dates_list[i]
        train_data_list, predict_data_list = [], []
        filter_index = pipeline_filter.read_filter_index(file_path=args.filter_file_path, period_index=i)
        for date in tqdm.tqdm(train_date_list[:10], desc="Loading training data"):
            file_path = os.path.join(args.data_dir, f"{date}.fea")
            data = pipeline_data.load_data(file_path)
            data.columns = ['code'] + [data.columns[j].strip() for j in range(len(data.columns) - 1)]   # Strip column names
            stock_code_col, feature_cols = data.columns[0], data.columns[3:15][filter_index]            # Select features based on filter index
            data = pd.concat([data[stock_code_col], data[feature_cols]], axis=1)                        # Keep only stock_code and feature columns
            data = pipeline_data.ensure_data_types(data)                                                # Ensure correct data types
            data = pipeline_data.fill_missing_values(data, fill_value=0.0)                              # Handle missing values
            data = pipeline_data.normalize_columns(data, feature_cols)                                  # Normalize features
            train_data_list.append(data)
        for date in tqdm.tqdm(predict_date_list[:10], desc="Loading prediction data"):
            file_path = os.path.join(args.data_dir, f"{date}.fea")
            data = pipeline_data.load_data(file_path)
            data.columns = ['code'] + [data.columns[j].strip() for j in range(len(data.columns) - 1)]   # Strip column names
            stock_code_col, feature_cols = data.columns[0], data.columns[3:15][filter_index]            # Select features based on filter index
            data = pd.concat([data[stock_code_col], data[feature_cols]], axis=1)                        # Keep only stock_code and feature columns
            data = pipeline_data.ensure_data_types(data)                                                # Ensure correct data types
            data = pipeline_data.fill_missing_values(data, fill_value=0.0)                              # Handle missing values
            data = pipeline_data.normalize_columns(data, feature_cols)                                  # Normalize features
            predict_data_list.append(pipeline_data.load_data(file_path))
        

    
    # data = pipeline_data.load_data('data/input_data.feather')
    
    # # Ensure correct data types
    # data = pipeline_data.ensure_data_types(data)
    
    # # Handle missing values
    # data = pipeline_data.fill_missing_values(data, fill_value=0.0)
    
    # # Normalize and standardize features
    # feature_columns = data.columns[1:]  # Assuming first column is 'stock_code'
    # data = pipeline_data.normalize_columns(data, feature_columns)
    # data = pipeline_data.standardize_columns(data, feature_columns)
    
    # # Filter features
    # label_column = 'target'  # Assuming there is a target column for prediction
    # data = pipeline_filter.select_features_by_ic_ir(data, label_column)
    
    # # Train model
    # model = pipeline_train.train_model(data, feature_columns.tolist(), label_column)
    
    # # Make predictions
    # predictions = pipeline_predict.make_predictions(model, data, feature_columns.tolist())
    
    # print(predictions)

main()
