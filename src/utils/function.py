from datetime import datetime

def generate_train_predict_dates(date_list, train_period_days=720, predict_period_days=60, slide_period_days=60, gap_days=10):
    """
    Generate training and prediction period date lists based on the provided trading dates.
    Period lengths and slide steps are measured in trading days (not calendar days).

    Parameters:
        date_list (list): List of trading dates in ascending order (format: 'YYYYMMDD')
        train_period_days (int): Number of trading days in each training period
        predict_period_days (int): Number of trading days in each prediction period
        slide_period_days (int): Number of trading days to slide the window forward each iteration
        gap_days (int): Number of trading days between the end of the training period and the start of the prediction period

    Returns:
        tuple: (len(train_dates_list), train_dates_list, predict_dates_list)
            - len(train_dates_list): Number of training/prediction periods
            - train_dates_list: List of all training periods, each as a sublist of dates
            - predict_dates_list: List of all prediction periods, each as a sublist of dates
    """
    train_dates_list = []
    predict_dates_list = []

    n = len(date_list)
    start_idx = 0

    while True:
        # Define index ranges
        train_start_idx = start_idx
        train_end_idx = start_idx + train_period_days
        predict_start_idx = train_end_idx + gap_days
        predict_end_idx = min(predict_start_idx + predict_period_days, n - 1)

        # Check boundary conditions
        if train_end_idx >= n - 1 - gap_days:
            break

        # Slice trading days based on indices
        train_dates = date_list[train_start_idx:train_end_idx]
        predict_dates = date_list[predict_start_idx:predict_end_idx]

        if not predict_dates:
            break

        train_dates_list.append(train_dates)
        predict_dates_list.append(predict_dates)

        # Slide window forward
        start_idx += slide_period_days

    return len(train_dates_list), train_dates_list, predict_dates_list
