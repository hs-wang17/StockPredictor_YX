import config.config_neural_network as config
import pipeline.filter as pipeline_filter
import pipeline.predict_neural_network_with_validation_update as pipeline_predict_neural_network
import utils.dataloader as utils_dataloader
import utils.function as utils_function
import utils.process_data_parallel as utils_process_data_parallel

import os
import torch


def run():
    # Setup logger
    args, logger = config.load_config_with_logger()

    # Generate training and prediction periods
    date_list = sorted([file_name[:8] for file_name in os.listdir(args.data_dir)])  # Get all available dates
    predict_date_list = [date for date in date_list if date >= args.start_date]  # Filter dates based on start and end dates

    logger.info(f"Predict Dates: {predict_date_list[0]} to {predict_date_list[-1]}; Length: {len(predict_date_list)} days")
    filter_index = pipeline_filter.read_filter_index(file_path=args.filter_file_path, period_index=0)

    # Loading and normalizing prediction data
    logger.info("Loading and normalizing prediction data...")
    predict_data_list, _ = utils_process_data_parallel.key_parallel(
        predict_date_list, args.data_dir, filter_index=filter_index, n_jobs_calc=args.n_jobs_calc, n_jobs_io=args.n_jobs_io, type="predict"
    )

    predict_dataset, _ = utils_dataloader.get_dataloader_predict(predict_data_list, batch_size=args.predict_batch_size, shuffle=False)
    model = [torch.load(model_path, weights_only=False) for model_path in utils_function.get_latest_model_paths(args.model_save_dir, args.k_folds)]
    _ = pipeline_predict_neural_network.make_predictions_neural_network(
        model,
        predict_dataset,
        logger,
        predictions_save_dir=args.predictions_save_dir,
        project_name=args.project_name,
        period_index=predict_date_list[-1],
        device=args.device,
    )

    logger.info("All periods processed.")


if __name__ == "__main__":
    run()
