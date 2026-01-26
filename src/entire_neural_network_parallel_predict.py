import config.config_neural_network as config
import pipeline.filter as pipeline_filter
import pipeline.predict_neural_network_with_validation_update as pipeline_predict_neural_network
import utils.dataloader as utils_dataloader
import utils.function as utils_function
import utils.process_data_parallel as utils_process_data_parallel

import os
import torch
import pandas as pd


def run():
    # Setup logger
    args, logger = config.load_config_with_logger()

    # Generate prediction periods
    date_list = sorted([file_name[:8] for file_name in os.listdir(args.data_dir)])  # Get all available dates
    date_list = [date for date in date_list if date >= args.start_date and date <= args.end_date]  # Filter dates based on start date

    num_periods, _, predict_dates_list = utils_function.generate_train_predict_dates(
        date_list,
        train_period_days=args.train_period_days,
        predict_period_days=args.predict_period_days,
        slide_period_days=args.slide_period_days,
        gap_days=args.gap_days,
        from_start=args.from_start,
        remove_abnormal=args.remove_abnormal,
    )

    logger.info(f"Number of periods: {num_periods}")
    logger.info("=" * 60)

    # Create predict_date_df directly
    predict_date_df_data = []
    for i in range(len(predict_dates_list)):
        for date in predict_dates_list[i]:
            predict_date_df_data.append({"date": date, "period": i})
    predict_date_df = pd.DataFrame(predict_date_df_data).set_index("date")

    logger.info(f"Predict Dates: {predict_date_df.index[0]} to {predict_date_df.index[-1]}; Length: {len(predict_date_df)} days")
    filter_index = pipeline_filter.read_filter_index(file_path=args.filter_file_path, period_index=0)

    # Loading and normalizing prediction data
    logger.info("Loading and normalizing prediction data...")
    predict_data_list, _ = utils_process_data_parallel.key_parallel(
        predict_date_df.index.tolist(), args.data_dir, filter_index=filter_index, n_jobs_calc=args.n_jobs_calc, n_jobs_io=args.n_jobs_io, type="predict"
    )

    predict_dataset, _ = utils_dataloader.get_dataloader_predict(predict_data_list, batch_size=args.predict_batch_size, shuffle=False)

    logger.info(f"Predict date df length: {len(predict_date_df)}")
    logger.info(f"Actual processed data length: {len(predict_data_list)}")

    # Make predictions
    all_model_paths = utils_function.get_all_model_paths(args.model_save_dir, args.k_folds)
    all_period_models = []
    for period_models_paths in all_model_paths:
        period_models = [torch.load(model_path, weights_only=False) for model_path in period_models_paths]
        all_period_models.append(period_models)

    _ = pipeline_predict_neural_network.make_all_period_predictions_neural_network(
        all_period_models,
        predict_dataset,
        logger,
        predictions_save_dir=args.predictions_save_dir,
        project_name=args.project_name,
        predict_date_df=predict_date_df,
        device=args.device,
    )

    logger.info("All periods processed.")


if __name__ == "__main__":
    run()
