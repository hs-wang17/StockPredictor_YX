import logging
import os
from datetime import datetime


def setup_logger(
    log_dir: str = "/home/user0/results/logs", log_filename: str = f"{datetime.now().strftime("%Y-%m-%d_%H%M%S")}.log", project_name: str = "StockPredictor"
) -> logging.Logger:
    """
    设置并返回一个日志记录器，日志将保存到指定文件并同时打印到控制台。

    Parameters:
        log_dir (str): 日志文件存储的目录。
        log_filename (str): 日志文件的名称。

    Returns:
        logging.Logger: 配置好的日志记录器。
    """
    # 如果日志目录不存在，创建它
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 日志文件的完整路径
    log_file = os.path.join(log_dir, f"{project_name}_{log_filename}")

    # 设置日志的基本配置
    logging.basicConfig(
        level=logging.DEBUG,  # 记录所有级别的日志
        format="%(asctime)s - %(levelname)s - %(message)s",  # 日志格式
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],  # 保存日志到文件  # 同时打印到控制台
    )

    # 返回日志记录器
    logger = logging.getLogger()
    logging.disable(logging.DEBUG)

    return logger
