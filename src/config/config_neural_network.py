import argparse
import os
import utils.logger as utils_logger


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the stock prediction pipeline.")

    parser.add_argument("--begin_period", type=int, default=0, help="Starting period for data processing (default: 0)")
    parser.add_argument("--criterion", type=str, default="mse", help="Loss function to use (default: 'mse')")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/haris/mydata/concat_daily_factor_with_label",
        help="Directory containing stock data files (default: '/home/haris/mydata/concat_daily_factor_with_label')",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for computation (default: 'cuda:0')")
    parser.add_argument("--end_date", type=str, default="20250930", help="End date for data processing (default: '20250930')")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs (default: 200)")
    parser.add_argument("--filter_file_path", type=str, default="config/filter_index.fea", help="Path to filter index file")
    parser.add_argument("--from_start", type=str2bool, default=False, help="Whether to train the model from scratch (default: False)")
    parser.add_argument("--gap_days", type=int, default=20, help="Days between end of training and start of prediction")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension size for the neural network (default: 64)")
    parser.add_argument("--inverse", type=str2bool, default=False, help="Whether to invert the training and prediction period (default: False)")
    parser.add_argument("--k_folds", type=int, default=4, help="Number of folds for K-fold cross-validation (default: 4)")
    # parser.add_argument("--label_file_path", type=str, default="/home/haris/mydata/label.fea", help="Path to label file")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training (default: 1e-4)")
    parser.add_argument("--lr_decay_gamma", type=float, default=0.99, help="Learning rate decay gamma (default: 0.99)")
    parser.add_argument("--log_dir", type=str, default="/home/haris/results/logs", help="Directory to save logs (default: '/home/haris/results/logs')")
    parser.add_argument("--model_type", type=str, default="mlp", help="Type of model to use (default: 'mlp')")
    parser.add_argument(
        "--model_save_dir", type=str, default="/home/haris/results/models", help="Directory to save models (default: '/home/haris/results/models')"
    )
    parser.add_argument("--num_periods", type=int, default=None, help="Number of periods to process (default: all)")
    parser.add_argument("--predict_batch_size", type=int, default=64, help="Batch size for prediction (default: 64)")
    parser.add_argument("--predict_period_days", type=int, default=60, help="Number of days for prediction period (default: 60)")
    parser.add_argument("--predictions_save_dir", type=str, default="/home/haris/results/predictions", help="Directory to save predictions")
    parser.add_argument("--project_name", type=str, default="StockPredictor", help="Name of the project/experiment")
    parser.add_argument("--model_save_frequency", type=int, default=5, help="Frequency (in epochs) to save model (default: 5)")
    parser.add_argument("--remove_abnormal", type=str2bool, default=True, help="Whether to remove abnormal data points (default: True)")
    parser.add_argument("--slide_period_days", type=int, default=60, help="Sliding window length in days (default: 60)")
    parser.add_argument("--start_date", type=str, default="20210101", help="Start date for data processing (default: '20200101')")
    parser.add_argument("--trade_date_path", type=str, default="/home/haris/mydata_20251231/trade_date.fea", help="Path to trade date file")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size for training (default: 1)")
    parser.add_argument("--train_period_days", type=int, default=720, help="Number of days for training period (default: 720)")
    parser.add_argument("--use_swanlab", type=str2bool, default=True, help="Enable SwanLab logging")
    parser.add_argument("--n_jobs_calc", type=int, default=64, help="Number of parallel jobs for calculation stage (default: 64)")
    parser.add_argument("--n_jobs_io", type=int, default=16, help="Number of parallel jobs for I/O stage (default: 16)")

    return parser.parse_args()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes", "y"):
        return True
    if v.lower() in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


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
