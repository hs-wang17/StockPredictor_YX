# 文件名: data_loader_parallel.py
import os
import sys
import time
import shutil
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm

# === 路径配置 ===
# 确保能导入您的 pipeline 模块
# 如果您的主程序已经添加了路径，这里其实可以省略，为了保险起见保留
sys.path.append("/home/haris/project/predictor/src")

try:
    import pipeline.data as pipeline_data
except ImportError:
    print("Error: 无法导入 pipeline.data，请检查路径配置")

# === 全局配置: 临时存储目录 ===
# 使用 /dev/shm (Linux 共享内存) 极大提升 IO 速度
TEMP_OUTPUT_DIR = "/dev/shm/nn_temp_processing_buffer"


def _process_single_file_worker(args_pack):
    """
    [内部Worker函数] 子进程执行的具体逻辑
    注意：此函数必须定义在模块顶层，否则 multiprocessing 无法序列化 (pickling)
    """
    date, data_dir, filter_index = args_pack

    file_path = os.path.join(data_dir, f"{date}.fea")
    output_path = os.path.join(TEMP_OUTPUT_DIR, f"proc_{date}.feather")

    try:
        # === 您的原始业务逻辑 ===
        data = pipeline_data.load_data(file_path)
        target = data["label"]
        data = data.drop(columns=["label"])
        data.columns = [data.columns[j].strip() for j in range(len(data.columns))]
        if filter_index is not None:
            feature_cols = data.columns[filter_index]
        else:
            feature_cols = data.columns
        data = pd.concat([data.index.to_frame(name="code"), data[feature_cols]], axis=1).reset_index(drop=True)
        data = pipeline_data.ensure_data_types(data)
        data = pipeline_data.fill_inf_with_nan(data)
        data = pipeline_data.winsorize_columns(data, feature_cols, lower_quantile=0.01, upper_quantile=0.99)
        data = pipeline_data.standardize_columns(data, feature_cols)
        data = pipeline_data.fill_missing_values(data)
        data = pd.concat([pd.DataFrame({"date": [date] * len(data)}), data], axis=1)
        data.to_feather(output_path)
        target.to_frame(name="target").to_feather(output_path.replace(".feather", "_target.feather"))
        return date, output_path, True, feature_cols

    except Exception as e:
        print(f"Error processing {date}: {e}")
        return date, None, False, None


def key_parallel(date_list, data_dir, filter_index=None, n_jobs_calc=128, n_jobs_io=32):
    """
    [对外接口] 并行加载数据的主函数

    Args:
        date_list: 日期字符串列表 (如 train_date_list)
        data_dir: 数据所在的文件夹路径
        filter_index: 特征筛选的索引 (可选)
        n_jobs_calc: 计算阶段并发数 (建议接近 CPU 核数)
        n_jobs_io: 加载阶段并发数 (建议 16-32)

    Returns:
        List[Tuple[pd.DataFrame, pd.Series]]: 处理好的数据和目标变量元组列表，按日期排序
    """
    # 1. 初始化环境
    if not os.path.exists(TEMP_OUTPUT_DIR):
        os.makedirs(TEMP_OUTPUT_DIR)

    print(f"[ParallelLoader] 开始处理 {len(date_list)} 天数据...")
    start_time = time.time()

    # 2. 准备参数
    args_list = [(date, data_dir, filter_index) for date in date_list]

    # 3. 第一阶段：多进程计算 (CPU Bound)
    # 使用 ProcessPoolExecutor 绕过 GIL 锁
    with ProcessPoolExecutor(max_workers=n_jobs_calc) as executor:
        results = list(tqdm(executor.map(_process_single_file_worker, args_list), total=len(args_list), desc="Stage 1: Calculation (CPU)"))

    print(f"[ParallelLoader] 计算完成，开始高速加载...")

    # 4. 排序结果 (确保时间顺序)
    results.sort(key=lambda x: x[0])
    valid_results = [(r[1], r[3]) for r in results if r[2] and r[1] and os.path.exists(r[1])]

    # 5. 第二阶段：多线程加载 (IO/Memory Bound)
    # 定义内部加载函数
    def _load_and_clean(args):
        data_path, feature_cols = args
        try:
            data = pd.read_feather(data_path)
            target = pd.read_feather(data_path.replace(".feather", "_target.feather"))["target"]
            os.remove(data_path)  # 读完即删
            os.remove(data_path.replace(".feather", "_target.feather"))
            return (data, target), feature_cols
        except:
            return None, None

    data_list = []
    feature_cols_list = None
    # 使用 ThreadPoolExecutor 共享内存读取
    with ThreadPoolExecutor(max_workers=n_jobs_io) as executor:
        results_loaded = list(tqdm(executor.map(_load_and_clean, valid_results), total=len(valid_results), desc="Stage 2: Loading (IO)"))
        for data_result, feature_cols in results_loaded:
            if data_result is not None:
                data_list.append(data_result)
                if feature_cols is not None and feature_cols_list is None:
                    feature_cols_list = feature_cols

    # 6. 清理残余文件
    try:
        shutil.rmtree(TEMP_OUTPUT_DIR)
    except:
        pass

    total_time = time.time() - start_time
    print(f"[ParallelLoader] 完成! 总耗时: {total_time:.2f}s, 数据集大小: {len(data_list)}")

    return data_list, feature_cols_list
