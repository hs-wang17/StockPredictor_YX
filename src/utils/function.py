import os
import re


def generate_train_predict_dates(
    date_list, train_period_days=720, predict_period_days=60, slide_period_days=60, gap_days=10, from_start=False, remove_abnormal=True
):
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
        if from_start:
            train_start_idx = 0
        else:
            train_start_idx = start_idx
        train_end_idx = start_idx + train_period_days
        predict_start_idx = train_end_idx + gap_days
        predict_end_idx = min(predict_start_idx + predict_period_days, n)

        # Check boundary conditions
        if train_end_idx >= n - gap_days:
            break

        # Slice trading days based on indices
        train_dates = date_list[train_start_idx:train_end_idx]
        if remove_abnormal:
            train_dates = [date for date in train_dates if (date < "20240201" or date > "20240223")]
        predict_dates = date_list[predict_start_idx:predict_end_idx]

        if not predict_dates:
            break

        train_dates_list.append(train_dates)
        predict_dates_list.append(predict_dates)

        # Slide window forward
        start_idx += slide_period_days

    return len(train_dates_list), train_dates_list, predict_dates_list


def get_latest_model_paths(model_save_dir, k_folds):
    """
    获取最后一个period中每个fold的最后一个epoch的模型文件路径

    Parameters:
        model_save_dir (str): 模型保存的根目录
        k_folds (int): fold的数量

    Returns:
        list: 包含k_folds个模型文件路径的列表，每个对应一个fold的最后一个epoch模型
    """

    model_paths = []

    # 找到所有包含period的目录
    all_dirs = [d for d in os.listdir(model_save_dir) if os.path.isdir(os.path.join(model_save_dir, d))]

    # 提取period编号，找到最后一个period
    period_numbers = []
    period_pattern = re.compile(r"period_(\d+)_fold\d+_model")

    for dir_name in all_dirs:
        match = period_pattern.search(dir_name)
        if match:
            period_numbers.append(int(match.group(1)))

    if not period_numbers:
        raise ValueError(f"在目录 {model_save_dir} 中没有找到符合格式的模型目录")

    latest_period = max(period_numbers)

    # 为每个fold找到对应的最后一个epoch模型
    for fold in range(k_folds):
        # 构建fold目录的匹配模式
        fold_dir_pattern = f"StockPredictor_*_period_{latest_period}_fold{fold}_model"

        # 找到匹配的fold目录
        fold_dirs = [d for d in all_dirs if re.match(fold_dir_pattern.replace("*", ".*").replace("_", "_"), d)]

        if not fold_dirs:
            raise ValueError(f"没有找到period {latest_period} fold {fold}的模型目录")

        # 使用第一个匹配的目录（通常只有一个）
        fold_dir = fold_dirs[0]
        fold_path = os.path.join(model_save_dir, fold_dir)

        # 找到该fold目录下的所有epoch文件
        epoch_files = [f for f in os.listdir(fold_path) if f.endswith(".pt")]

        if not epoch_files:
            raise ValueError(f"在目录 {fold_path} 中没有找到.pt文件")

        # 提取epoch编号，找到最后一个
        epoch_pattern = re.compile(r"epoch(\d+)\.pt")
        epoch_numbers = []

        for epoch_file in epoch_files:
            match = epoch_pattern.search(epoch_file)
            if match:
                epoch_numbers.append(int(match.group(1)))

        if not epoch_numbers:
            raise ValueError(f"在目录 {fold_path} 中没有找到符合格式的epoch文件")

        latest_epoch = max(epoch_numbers)

        # 找到具体的文件名
        model_file = None
        for epoch_file in epoch_files:
            if f"epoch{latest_epoch}.pt" in epoch_file:
                model_file = epoch_file
                break

        if not model_file:
            raise ValueError(f"没有找到epoch {latest_epoch}的模型文件")

        model_path = os.path.join(fold_path, model_file)
        model_paths.append(model_path)

    return model_paths


def get_all_model_paths(model_save_dir, k_folds):
    """
    获取所有period中每个fold的最后一个epoch的模型文件路径

    Parameters:
        model_save_dir (str): 模型保存的根目录
        k_folds (int): fold的数量

    Returns:
        list: 包含(num_period, k_folds)个模型文件路径的二维列表，每个对应一个fold的最后一个epoch模型
              结构为 [[period0_fold0, period0_fold1, ...], [period1_fold0, period1_fold1, ...], ...]
    """

    # 找到所有包含period的目录
    all_dirs = [d for d in os.listdir(model_save_dir) if os.path.isdir(os.path.join(model_save_dir, d))]

    # 提取所有period编号
    period_numbers = []
    period_pattern = re.compile(r"period_(\d+)_fold\d+_model")

    for dir_name in all_dirs:
        match = period_pattern.search(dir_name)
        if match:
            period_numbers.append(int(match.group(1)))

    if not period_numbers:
        raise ValueError(f"在目录 {model_save_dir} 中没有找到符合格式的模型目录")

    # 获取所有唯一的period编号并排序
    unique_periods = sorted(list(set(period_numbers)))

    # 初始化二维列表：行是period，列是fold
    model_paths = []

    # 对每个period进行处理
    for period in unique_periods:
        period_models = []

        # 为当前period的每个fold找到最后一个epoch模型
        for fold in range(k_folds):
            # 构建fold目录的匹配模式
            fold_dir_pattern = f"StockPredictor_*_period_{period}_fold{fold}_model"

            # 找到匹配的fold目录
            fold_dirs = [d for d in all_dirs if re.match(fold_dir_pattern.replace("*", ".*").replace("_", "_"), d)]

            if not fold_dirs:
                raise ValueError(f"没有找到period {period} fold {fold}的模型目录")

            # 使用第一个匹配的目录（通常只有一个）
            fold_dir = fold_dirs[0]
            fold_path = os.path.join(model_save_dir, fold_dir)

            # 找到该fold目录下的所有epoch文件
            epoch_files = [f for f in os.listdir(fold_path) if f.endswith(".pt")]

            if not epoch_files:
                raise ValueError(f"在目录 {fold_path} 中没有找到.pt文件")

            # 提取epoch编号，找到最后一个
            epoch_pattern = re.compile(r"epoch(\d+)\.pt")
            epoch_numbers = []

            for epoch_file in epoch_files:
                match = epoch_pattern.search(epoch_file)
                if match:
                    epoch_numbers.append(int(match.group(1)))

            if not epoch_numbers:
                raise ValueError(f"在目录 {fold_path} 中没有找到符合格式的epoch文件")

            latest_epoch = max(epoch_numbers)

            # 找到具体的模型文件名
            model_file = None
            for epoch_file in epoch_files:
                if f"epoch{latest_epoch}.pt" in epoch_file:
                    model_file = epoch_file
                    break

            if not model_file:
                raise ValueError(f"没有找到epoch {latest_epoch}的模型文件")

            model_path = os.path.join(fold_path, model_file)
            period_models.append(model_path)

        # 将当前period的所有fold模型添加到结果中
        model_paths.append(period_models)

    return model_paths
