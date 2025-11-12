import config.config as config
import pipeline.data as pipeline_data
import pipeline.filter as pipeline_filter
import pipeline.predict_neural_network as pipeline_predict_neural_network
import pipeline.train_neural_network as pipeline_train_neural_network
import utils.dataloader as utils_dataloader
import utils.function as utils_function
import utils.logger as utils_logger
import utils.neural_network_model as utils_neural_network_model

import numpy as np
import pandas as pd
import os
import tqdm
import swanlab

def run():
    # Setup logger
    args, logger = config.load_config_with_logger()

    # Generate training and prediction periods
    date_list = sorted([file_name[:8] for file_name in os.listdir(args.data_dir)])
    num_periods, train_dates_list, predict_dates_list = utils_function.generate_train_predict_dates(
        date_list, 
        train_period_days=args.train_period_days, 
        predict_period_days=args.predict_period_days, 
        slide_period_days=args.slide_period_days,
        gap_days=args.gap_days
    )

    logger.info(f"Number of periods: {num_periods}")
    logger.info("=" * 60)

    all_predictions_list = []

    for i in range(num_periods)[:2]:
        logger.info(f"Period {i+1}:")
        logger.info(f"  Train Dates: {train_dates_list[i][0]} to {train_dates_list[i][-1]}; Length: {len(train_dates_list[i])} days")
        logger.info(f"  Predict Dates: {predict_dates_list[i][0]} to {predict_dates_list[i][-1]}; Length: {len(predict_dates_list[i])} days")

        # Load and preprocess data for the current period
        train_date_list, predict_date_list = train_dates_list[i], predict_dates_list[i]
        train_data_list, predict_data_list = [], []
        filter_index = pipeline_filter.read_filter_index(file_path=args.filter_file_path, period_index=i)

        for date in tqdm.tqdm(train_date_list[:100], desc="Loading training data"):
            file_path = os.path.join(args.data_dir, f"{date}.fea")
            data = pipeline_data.load_data(file_path)
            data.columns = ['code'] + [data.columns[j].strip() for j in range(len(data.columns) - 1)]   # Strip column names
            stock_code_col, feature_cols = data.columns[0], data.columns[3:15][filter_index]            # Select features based on filter index
            data = pd.concat([data[stock_code_col], data[feature_cols]], axis=1)                        # Keep only stock_code and feature columns
            data = pipeline_data.ensure_data_types(data)                                                # Ensure correct data types
            data = pipeline_data.fill_missing_values(data, fill_value=0.0)                              # Handle missing values
            data = pipeline_data.normalize_columns(data, feature_cols)                                  # Normalize features
            data.insert(0, 'date', date)                                                                # Add date column
            target = pipeline_data.load_data(file_path).iloc[:, 4]                                      # TODO: Adjust target column as needed
            train_data_list.append((data, target))

        for date in tqdm.tqdm(predict_date_list[:100], desc="Loading prediction data"):
            file_path = os.path.join(args.data_dir, f"{date}.fea")
            data = pipeline_data.load_data(file_path)
            data.columns = ['code'] + [data.columns[j].strip() for j in range(len(data.columns) - 1)]   # Strip column names
            stock_code_col, feature_cols = data.columns[0], data.columns[3:15][filter_index]            # Select features based on filter index
            data = pd.concat([data[stock_code_col], data[feature_cols]], axis=1)                        # Keep only stock_code and feature columns
            data = pipeline_data.ensure_data_types(data)                                                # Ensure correct data types
            data = pipeline_data.fill_missing_values(data, fill_value=0.0)                              # Handle missing values
            data = pipeline_data.normalize_columns(data, feature_cols)                                  # Normalize features
            data.insert(0, 'date', date)                                                                # Add date column
            target = pipeline_data.load_data(file_path).iloc[:, 4]                                      # TODO: Adjust target column as needed
            predict_data_list.append((data, target))

        train_dataloader = utils_dataloader.get_dataloader(train_data_list, batch_size=args.train_batch_size, shuffle=False)
        predict_dataloader = utils_dataloader.get_dataloader(predict_data_list, batch_size=args.predict_batch_size, shuffle=False)

        # Train model
        model = utils_neural_network_model.neural_network_model(input_dim=len(feature_cols), hidden_dim=64, output_dim=1)
        model = pipeline_train_neural_network.train_neural_network_model(
            model, train_dataloader, logger, 
            model_save_dir=args.model_save_dir, 
            epochs=args.epochs, 
            learning_rate=args.learning_rate,
            device=args.device,
            project_name=args.project_name,
            period_index=i,
            model_save_frequency=args.model_save_frequency,
            use_swanlab=args.use_swanlab
            )
        
        # Make predictions
        predictions = pipeline_predict_neural_network.make_predictions_neural_network(
            model, predict_dataloader, logger,
            predictions_save_dir=args.predictions_save_dir,
            device=args.device,
            use_swanlab=args.use_swanlab
            )
        
        all_predictions_list.append(predictions)

    # Combine all period predictions
    logger.info("Concatenating all period predictions...")
    combined_predictions = pd.concat(all_predictions_list)
    combined_output_path = os.path.join(args.predictions_save_dir, f"{args.project_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_combined_predictions.csv")
    combined_predictions.to_csv(combined_output_path)
    logger.info(f"All periods combined predictions saved to {combined_output_path}")
    logger.info("All periods processed.")

    if args.use_swanlab:
        swanlab.finish()

if __name__ == "__main__":
    run()
