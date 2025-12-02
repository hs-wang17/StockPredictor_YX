import argparse
import os
import utils.logger as utils_logger


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the stock prediction pipeline.")

    parser.add_argument("--begin_period", type=int, default=0, help="Starting period for data processing (default: 0)")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/user0/mydata/concat_daily_factor_with_label",
        help="Directory containing stock data files (default: '/home/user0/mydata/concat_daily_factor_with_label')",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for computation ('cuda' or 'cpu')")
    parser.add_argument("--end_date", type=str, default="20250930", help="End date for data processing (default: '20250930')")
    parser.add_argument("--filter_file_path", type=str, default="config/filter_index.fea", help="Path to filter index file")
    parser.add_argument("--gap_days", type=int, default=20, help="Days between end of training and start of prediction")
    parser.add_argument("--k_folds", type=int, default=4, help="Number of folds for K-fold cross-validation (default: 4)")
    parser.add_argument("--label_file_path", type=str, default="/home/user0/mydata/label.fea", help="Path to label file")
    parser.add_argument("--log_dir", type=str, default="/home/user0/results/logs", help="Directory to save logs")
    parser.add_argument("--model_save_dir", type=str, default="/home/user0/results/models", help="Directory to save models")
    parser.add_argument("--num_periods", type=int, default=None, help="Number of periods to process (default: all)")
    parser.add_argument("--predict_period_days", type=int, default=60, help="Number of days for prediction period (default: 60)")
    parser.add_argument("--predictions_save_dir", type=str, default="/home/user0/results/predictions", help="Directory to save predictions")
    parser.add_argument("--project_name", type=str, default="StockPredictor", help="Name of the project/experiment")
    parser.add_argument("--slide_period_days", type=int, default=60, help="Sliding window length in days (default: 60)")
    parser.add_argument("--start_date", type=str, default="20210101", help="Start date for data processing (default: '20200101')")
    parser.add_argument("--train_period_days", type=int, default=720, help="Number of days for training period (default: 720)")
    parser.add_argument("--use_swanlab", type=bool, default=True, help="Enable SwanLab logging")
    parser.add_argument("--use_gpu", type=bool, default=True, help="Use GPU for training if available (default: True)")

    parser.add_argument("--n_estimators", type=int, default=1000, help="Number of boosting iterations for LightGBM (default: 1000)")
    parser.add_argument("--objective", type=str, default="regression", help="Objective function for LightGBM (default: 'regression')")
    parser.add_argument("--boosting_type", type=str, default="gbdt", help="Boosting type for LightGBM (default: 'gbdt')")
    parser.add_argument("--random_state", type=int, default=42, help="Random random_state for reproducibility (default: 42)")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate for LightGBM (default: 0.1)")
    parser.add_argument("--num_leaves", type=int, default=31, help="Maximum number of leaves for LightGBM (default: 31)")
    parser.add_argument("--min_data_in_leaf", type=int, default=20, help="Minimum data in leaf for LightGBM (default: 20)")
    parser.add_argument("--feature_fraction", type=float, default=0.8, help="Feature fraction for LightGBM (default: 0.8)")
    parser.add_argument("--bagging_fraction", type=float, default=0.8, help="Bagging fraction for LightGBM (default: 0.8)")
    parser.add_argument("--bagging_freq", type=int, default=1, help="Bagging frequency for LightGBM (default: 1)")
    parser.add_argument("--early_stopping_rounds", type=int, default=50, help="Early stopping rounds for LightGBM (default: 50)")
    parser.add_argument("--valid_size", type=float, default=0.1, help="Validation set proportion (default: 0.1)")
    parser.add_argument("--verbose_eval", type=int, default=50, help="LightGBM evaluation log frequency (default: 50)")

    return parser.parse_args()


def ensure_directories(args):
    """Ensure that all output directories exist."""
    for path in [args.log_dir, args.model_save_dir, args.predictions_save_dir]:
        os.makedirs(path, exist_ok=True)


def show_config(args, logger=None):
    """Pretty-print all configuration parameters."""
    if logger:
        logger.info("=" * 60)
        logger.info("Experiment Configuration")
        logger.info("=" * 60)
        for k, v in vars(args).items():
            logger.info(f"{k:25s}: {v}")
        logger.info("=" * 60)
    else:
        print("=" * 60)
        print("Experiment Configuration")
        print("=" * 60)
        for k, v in vars(args).items():
            print(f"{k:25s}: {v}")
        print("=" * 60)


def load_config_with_logger():
    args = parse_args()
    ensure_directories(args)
    logger = utils_logger.setup_logger(log_dir=args.log_dir)
    show_config(args, logger=logger)
    return args, logger
