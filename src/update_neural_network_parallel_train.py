import config.config_neural_network as config
import pipeline.filter as pipeline_filter
import pipeline.train_neural_network_with_validation_parallel as pipeline_train_neural_network
import utils.dataloader as utils_dataloader
import utils.function as utils_function
import utils.process_data_parallel as utils_process_data_parallel

import pandas as pd
import os
from datetime import datetime


def run():
    # Setup logger
    args, logger = config.load_config_with_logger()

    # Generate training and prediction periods
    date_list = sorted([file_name[:8] for file_name in os.listdir(args.data_dir)])  # Get all available dates
    trade_date_list = pd.read_feather(args.trade_date_path)  # Load trade date list
    date_list = [date for date in date_list if date in trade_date_list["trade_date"].astype(str).tolist()]  # Filter dates to include only trade dates
    date_list = [date for date in date_list if date >= args.start_date]  # Filter dates based on start and end dates

    if (len(date_list) - (args.train_period_days + args.gap_days)) % args.slide_period_days != 1:
        logger.info(
            f"No need to update the model. Exiting days ({(len(date_list) - (args.train_period_days + args.gap_days)) % args.slide_period_days}/{args.slide_period_days})."
        )

    else:
        logger.info(f"Need to update the model. Proceeding with training.")

        num_periods, train_dates_list, _ = utils_function.generate_train_predict_dates(
            date_list, train_period_days=args.train_period_days, slide_period_days=args.slide_period_days, remove_abnormal=args.remove_abnormal
        )  # Generate train date lists for each period

        logger.info(f"Number of periods: {num_periods}")
        logger.info("=" * 60)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i in range(num_periods)[:1]:  # only update the last period
            if args.inverse:
                i = num_periods - 1 - i
            if hasattr(args, "begin_period") and i < args.begin_period:
                logger.info(f"Skipping period {i} because begin_period={args.begin_period}")
                continue

            logger.info(f"Period {i}:")
            logger.info(f"  Train Dates: {train_dates_list[i][0]} to {train_dates_list[i][-1]}; Length: {len(train_dates_list[i])} days")

            train_date_list = train_dates_list[i]
            filter_index = pipeline_filter.read_filter_index(file_path=args.filter_file_path, period_index=i)

            # Loading and preprocessing training data
            logger.info("Loading and preprocessing training data...")
            train_data_list, feature_cols = utils_process_data_parallel.key_parallel(
                train_date_list, args.data_dir, filter_index=filter_index, n_jobs_calc=args.n_jobs_calc, n_jobs_io=args.n_jobs_io, type="train"
            )

            train_dataset, _ = utils_dataloader.get_dataloader(train_data_list, batch_size=args.train_batch_size, shuffle=False)

            # Train model
            model_param_dict = {"input_dim": len(feature_cols), "hidden_dim": args.hidden_dim, "output_dim": 1, "model_type": args.model_type}
            _ = pipeline_train_neural_network.train_neural_network_model_parallel(
                model_param_dict,
                train_dataset,  # train_dataloader when without validation
                logger,
                model_save_dir=args.model_save_dir,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                project_name=args.project_name,
                period_index=i,
                model_save_frequency=args.model_save_frequency,
                use_swanlab=args.use_swanlab,
                k_folds=args.k_folds,
                lr_decay_gamma=args.lr_decay_gamma,
                batch_size=args.train_batch_size,
                timestamp=timestamp,
            )

        logger.info("All periods processed.")


if __name__ == "__main__":
    run()
