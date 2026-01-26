import os
import pandas as pd
import swanlab
from datetime import datetime
import config.config_neural_network as config
import pipeline.filter as pipeline_filter
import pipeline.predict_neural_network_with_validation as pipeline_predict_neural_network
import pipeline.train_neural_network_with_validation_parallel as pipeline_train_neural_network
import utils.dataloader as utils_dataloader
import utils.function as utils_function
import utils.process_data_parallel as utils_process_data_parallel


def run():
    # Setup logger
    args, logger = config.load_config_with_logger()

    # Generate training and prediction periods
    date_list = sorted([file_name[:8] for file_name in os.listdir(args.data_dir)[:]])  # Get all available dates
    trade_date_list = pd.read_feather(args.trade_date_path)  # Load trade date list
    date_list = [date for date in date_list if date in trade_date_list["trade_date"].astype(str).tolist()]  # Filter dates to include only trade dates
    date_list = [date for date in date_list if date >= args.start_date and date <= args.end_date]  # Filter dates based on start and end dates

    num_periods, train_dates_list, predict_dates_list = utils_function.generate_train_predict_dates(
        date_list,
        train_period_days=args.train_period_days,
        predict_period_days=args.predict_period_days,
        slide_period_days=args.slide_period_days,
        gap_days=args.gap_days,
        from_start=args.from_start,
        remove_abnormal=args.remove_abnormal,
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
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "lr_decay_gamma": args.lr_decay_gamma,
                "hidden_dim": args.hidden_dim,
                "device": args.device,
                "train_batch_size": args.train_batch_size,
                "predict_batch_size": args.predict_batch_size,
                "train_period_days": args.train_period_days,
                "predict_period_days": args.predict_period_days,
                "slide_period_days": args.slide_period_days,
                "gap_days": args.gap_days,
                "model_save_frequency": args.model_save_frequency,
                "k_folds": args.k_folds,
            },
        )

    for i in range(num_periods):
        if args.inverse:
            i = num_periods - 1 - i
        if hasattr(args, "begin_period") and i < args.begin_period:
            logger.info(f"Skipping period {i} because begin_period={args.begin_period}")
            continue

        logger.info(f"Period {i}:")
        logger.info(f"  Train Dates: {train_dates_list[i][0]} to {train_dates_list[i][-1]}; Length: {len(train_dates_list[i])} days")
        logger.info(f"  Predict Dates: {predict_dates_list[i][0]} to {predict_dates_list[i][-1]}; Length: {len(predict_dates_list[i])} days")

        train_date_list, predict_date_list = train_dates_list[i], predict_dates_list[i]
        train_data_list, predict_data_list = [], []
        filter_index = pipeline_filter.read_filter_index(file_path=args.filter_file_path, period_index=i)

        # Loading and preprocessing training data using parallel processing
        logger.info("Loading and preprocessing training data...")
        train_data_list, feature_cols = utils_process_data_parallel.key_parallel(
            train_date_list, args.data_dir, filter_index=filter_index, n_jobs_calc=args.n_jobs_calc, n_jobs_io=args.n_jobs_io, type="train"
        )

        # Loading and preprocessing prediction data using parallel processing
        logger.info("Loading and preprocessing prediction data...")
        predict_data_list, _ = utils_process_data_parallel.key_parallel(
            predict_date_list, args.data_dir, filter_index=filter_index, n_jobs_calc=args.n_jobs_calc, n_jobs_io=args.n_jobs_io, type="predict"
        )

        train_dataset, _ = utils_dataloader.get_dataloader(train_data_list, batch_size=args.train_batch_size, shuffle=False)
        predict_dataset, _ = utils_dataloader.get_dataloader_predict(predict_data_list, batch_size=args.predict_batch_size, shuffle=False)

        # Train model
        model_param_dict = {"input_dim": len(feature_cols), "hidden_dim": args.hidden_dim, "output_dim": 1, "model_type": args.model_type}
        models = pipeline_train_neural_network.train_neural_network_model_parallel(
            model_param_dict=model_param_dict,
            dataset=train_dataset,
            logger=logger,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            criterion=args.criterion,
            model_save_dir=args.model_save_dir,
            save_model=True,
            project_name=args.project_name,
            period_index=i,
            model_save_frequency=args.model_save_frequency,
            use_swanlab=args.use_swanlab,
            k_folds=args.k_folds,
            lr_decay_gamma=args.lr_decay_gamma,
            batch_size=args.train_batch_size,
            timestamp=timestamp,
        )

        # Make predictions
        predictions = pipeline_predict_neural_network.make_predictions_neural_network(
            models=models,
            dataset=predict_dataset,
            logger=logger,
            predictions_save_dir=args.predictions_save_dir,
            project_name=args.project_name,
            device=args.device,
            use_swanlab=args.use_swanlab,
            period_index=i,
            model_save_dir=args.model_save_dir,
            timestamp=timestamp,
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
