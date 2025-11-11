import argparse
import os


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the stock prediction pipeline.")

    parser.add_argument(
        '--data_dir', type=str, default='/home/user0/data/StockDailyData/',
        help="Directory containing stock data files"
    )
    parser.add_argument(
        '--filter_file_path', type=str, default='config/filter_index.fea',
        help="Path to filter index file"
    )
    parser.add_argument(
        '--log_dir', type=str, default='/home/user0/results/logs',
        help="Directory to save logs"
    )
    parser.add_argument(
        '--model_dir', type=str, default='/home/user0/results/models',
        help="Directory to save models"
    )
    parser.add_argument(
        '--num_periods', type=int, default=None,
        help="Number of periods to process (default: all)"
    )
    parser.add_argument(
        '--predict_batch_size', type=int, default=1,
        help="Batch size for model prediction (default: 64)"
    )
    parser.add_argument(
        '--predict_period_months', type=int, default=3,
        help="Number of months for the prediction period (default: 3)"
    )
    parser.add_argument(
        '--results_dir', type=str, default='/home/user0/results/predictions',
        help="Directory to save predictions"
    )
    parser.add_argument(
        '--slide_period_months', type=int, default=3,
        help="Number of months for the sliding window (default: 3)"
    )
    parser.add_argument(
        '--train_batch_size', type=int, default=1,
        help="Batch size for model training (default: 64)"
    )
    parser.add_argument(
        '--train_period_years', type=int, default=3,
        help="Number of years for the training period (default: 3)"
    )

    return parser.parse_args()


def ensure_directories(args):
    """Ensure that all output directories exist."""
    for path in [args.log_dir, args.model_dir, args.results_dir]:
        os.makedirs(path, exist_ok=True)


def show_config(args):
    """Pretty-print all configuration parameters."""
    print("=" * 60)
    print("Experiment Configuration")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"{k:25s}: {v}")
    print("=" * 60)


def load_config():
    """
    Unified entry point:
    - Parse arguments
    - Create necessary directories
    - Display current configuration
    """
    args = parse_args()
    ensure_directories(args)
    show_config(args)
    
    return args
