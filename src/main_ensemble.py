import config.config_ensemble as config
import pipeline.data as pipeline_data
import pipeline.filter as pipeline_filter
import pipeline.predict_ensemble_with_validation as pipeline_predict_ensemble
import pipeline.train_ensemble_with_validation_parallel as pipeline_train_ensemble
import utils.ensemble_model as utils_ensemble_model
import utils.function as utils_function
import utils.logger as utils_logger

import numpy as np
import pandas as pd
import os
import tqdm
import swanlab
from datetime import datetime


def run():
    # Setup logger
    args, logger = config.load_config_with_logger()

    # Generate training and prediction periods
    date_list = sorted([file_name[:8] for file_name in os.listdir(args.data_dir)[:]])  # Get all available dates
    trade_date_list = pd.read_feather("/home/haris/mydata/trade_date.fea")  # Load trade date list
    date_list = [date for date in date_list if date in trade_date_list["trade_date"].astype(str).tolist()]  # Filter dates to include only trade dates
    date_list = [date for date in date_list if date >= args.start_date and date <= args.end_date]  # Filter dates based on start and end dates

    num_periods, train_dates_list, predict_dates_list = utils_function.generate_train_predict_dates(
        date_list,
        train_period_days=args.train_period_days,
        predict_period_days=args.predict_period_days,
        slide_period_days=args.slide_period_days,
        gap_days=args.gap_days,
        from_start=args.from_start,
    )  # Generate train and predict date lists for each period

    logger.info(f"Number of periods: {num_periods}")
    logger.info("=" * 60)

    # label = pipeline_data.load_data(args.label_file_path)
    all_predictions_list = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Initialize SwanLab if enabled
    if args.use_swanlab:
        swanlab.init(
            project=args.project_name,
            experiment_name=f"{timestamp}",
            log_level="ERROR",
            config={
                "train_period_days": args.train_period_days,
                "predict_period_days": args.predict_period_days,
                "slide_period_days": args.slide_period_days,
                "gap_days": args.gap_days,
                "n_estimators": args.n_estimators,
                "early_stopping_rounds": args.early_stopping_rounds,
                "valid_size": args.valid_size,
                "random_state": args.random_state,
            },
        )

    for i in range(num_periods):
        if hasattr(args, "begin_period") and i < args.begin_period:
            logger.info(f"Skipping period {i+1} because begin_period={args.begin_period}")
            continue

        logger.info(f"Period {i+1}:")
        logger.info(f"  Train Dates: {train_dates_list[i][0]} to {train_dates_list[i][-1]}; Length: {len(train_dates_list[i])} days")
        logger.info(f"  Predict Dates: {predict_dates_list[i][0]} to {predict_dates_list[i][-1]}; Length: {len(predict_dates_list[i])} days")

        # Load and preprocess data for the current period
        train_date_list, predict_date_list = train_dates_list[i], predict_dates_list[i]
        train_frames, predict_frames = [], []
        filter_index = pipeline_filter.read_filter_index(file_path=args.filter_file_path, period_index=i)

        # Loading and preprocessing training data
        logger.info("Loading and normalizing training data...")
        for date in tqdm.tqdm(train_date_list, desc="Loading training data"):
            file_path = os.path.join(args.data_dir, f"{date}.fea")
            data = pipeline_data.load_data(file_path)
            target = data["label"]
            data = data.drop(columns=["label"])
            data.columns = [data.columns[j].strip() for j in range(len(data.columns))]  # Strip column names
            if filter_index is not None:
                feature_cols = data.columns[filter_index]  # Select features based on filter index
            else:
                feature_cols = data.columns
            data = pd.concat([data.index.to_frame(name="code"), data[feature_cols]], axis=1).reset_index(drop=True)  # Keep only stock_code and feature columns
            data = pipeline_data.ensure_data_types(data)  # Ensure correct data types
            data = pipeline_data.fill_inf_with_nan(data)  # Handle infinite values
            data = pipeline_data.winsorize_columns(data, feature_cols, lower_quantile=0.01, upper_quantile=0.99)
            data = pipeline_data.standardize_columns(data, feature_cols)
            data = pd.concat([pd.DataFrame({"date": [date] * len(data)}), data], axis=1)
            temp = data.copy()
            temp["target"] = target.values
            temp = pipeline_data.fill_missing_values(temp)
            train_frames.append(temp)

        # Loading and normalizing prediction data
        logger.info("Loading and normalizing prediction data...")
        for date in tqdm.tqdm(predict_date_list, desc="Loading prediction data"):
            file_path = os.path.join(args.data_dir, f"{date}.fea")
            data = pipeline_data.load_data(file_path)
            target = data["label"]
            data = data.drop(columns=["label"])
            data.columns = [data.columns[j].strip() for j in range(len(data.columns))]  # Strip column names
            if filter_index is not None:
                feature_cols = data.columns[filter_index]  # Select features based on filter index
            else:
                feature_cols = data.columns
            data = pd.concat([data.index.to_frame(name="code"), data[feature_cols]], axis=1).reset_index(drop=True)  # Keep only stock_code and feature columns
            data = pipeline_data.ensure_data_types(data)  # Ensure correct data types
            data = pipeline_data.fill_inf_with_nan(data)  # Handle infinite values
            data = pipeline_data.winsorize_columns(data, feature_cols, lower_quantile=0.01, upper_quantile=0.99)
            data = pipeline_data.standardize_columns(data, feature_cols)
            data = pd.concat([pd.DataFrame({"date": [date] * len(data)}), data], axis=1)
            temp = data.copy()
            temp["target"] = target.values
            temp = pipeline_data.fill_missing_values(temp)
            predict_frames.append(temp)

        train_df = pd.concat(train_frames, ignore_index=True)
        predict_df = pd.concat(predict_frames, ignore_index=True)

        # Create LightGBM model
        base_model = utils_ensemble_model.create_lgbm_model(
            n_estimators=args.n_estimators,
            objective=args.objective,
            boosting_type=args.boosting_type,
            random_state=args.random_state,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            min_data_in_leaf=args.min_data_in_leaf,
            feature_fraction=args.feature_fraction,
            bagging_fraction=args.bagging_fraction,
            bagging_freq=args.bagging_freq,
        )

        # Train model
        models = pipeline_train_ensemble.train_lightgbm_model(
            model=base_model,
            train_df=train_df,
            logger=logger,
            model_save_dir=args.model_save_dir,
            period_index=i,
            project_name=args.project_name,
            k_folds=args.k_folds,
            early_stopping_rounds=args.early_stopping_rounds,
            verbose_eval=args.verbose_eval,
            random_state=args.random_state,
            use_gpu=args.use_gpu,
            timestamp=timestamp,
            feature_cols=feature_cols,
        )

        # Make predictions
        predictions = pipeline_predict_ensemble.make_predictions_lightgbm(
            models=models,
            predict_df=predict_df,
            logger=logger,
            predictions_save_dir=args.predictions_save_dir,
            project_name=args.project_name,
            period_index=i,
            date_col="date",
            code_col="code",
            timestamp=timestamp,
            feature_cols=feature_cols,
        )
        all_predictions_list.append(predictions)

    # Combine all period predictions
    logger.info("Concatenating all period predictions...")
    combined_predictions = pd.concat(all_predictions_list, axis=1, join="outer")
    combined_output_path = os.path.join(
        args.predictions_save_dir, f"{args.project_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_combined_predictions.csv"
    )
    combined_predictions.to_csv(combined_output_path)
    logger.info(f"All periods combined predictions saved to {combined_output_path}")
    logger.info("All periods processed.")

    if args.use_swanlab:
        swanlab.finish()


if __name__ == "__main__":
    run()
