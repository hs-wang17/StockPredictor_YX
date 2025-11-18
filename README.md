# 🚀 StockPredictor

## 📊 Overview / 项目概述

**StockPredictor** is a Python-based stock prediction pipeline designed to train models on historical trading data and generate predictions for future stock prices. It supports both **Neural Networks** and **LightGBM** models.
**StockPredictor** 是一个基于 Python 的股票预测管道，专注于使用历史交易数据训练模型，并生成未来股票价格预测。它同时支持 **神经网络（Neural Networks）** 和 **集成（LightGBM）** 两种模型。

The pipeline integrates:

- Data loading & preprocessing / 数据加载与预处理
- Feature selection & filtering / 特征选择与过滤
- Model training & prediction / 模型训练与预测
- Logging & configurable parameters / 日志记录与可配置参数

---

## 📁 Directory Structure / 目录结构

```
predictor/
├── README.md                          # Project overview & usage instructions / 项目概述与使用说明
├── docs/                              # Documentation / 文档
│   └── flowchart.md                   # Workflow flowchart / 流程图
├── scripts/                           # Auxiliary scripts / 辅助脚本
└── src/                               # Source code / 源代码
    ├── main.py                        # Main entry point / 主入口
    ├── pipeline/                      # Core pipeline modules / 核心流水线模块
    │   ├── data.py                    # Data loading & preprocessing / 数据加载与预处理
    │   ├── filter.py                  # Feature selection & filtering / 特征选择与过滤
    │   ├── train_neural_network.py    # NN training module / 神经网络训练模块
    │   ├── predict_neural_network.py  # NN prediction module / 神经网络预测模块
    │   ├── train_ensemble.py          # LightGBM / Ensemble training / LightGBM训练模块
    │   └── predict_ensemble.py        # LightGBM prediction module / LightGBM预测模块
    └── utils/                         # Utility functions / 工具函数
        ├── function.py                # General helpers / 通用辅助函数
        ├── logger.py                  # Logger setup / 日志工具
        ├── dataloader.py              # DataLoader utilities (for NN) / 神经网络 DataLoader 工具
        └── model.py                   # Model definitions / 模型定义
```

---

## 🛠️ Installation / 安装

1. Clone the repository / 克隆仓库：

```bash
git clone https://github.com/yourusername/predictor.git
cd predictor
```

2. Create a Python virtual environment / 创建虚拟环境：

```bash
python -m venv myenv
source myenv/bin/activate   # Linux/macOS
myenv\Scripts\activate      # Windows
```

3. Install dependencies / 安装依赖：

```bash
pip install -r requirements.txt
```

Typical dependencies / 主要依赖：

- `torch` 🧠
- `pandas` 🐼
- `numpy` 🔢
- `tqdm` ⏳
- `lightgbm` 🌳

---

## ⚙️ Configuration / 配置

All parameters are configurable via command-line arguments / 所有参数均可通过命令行指定。

| Argument                  | Description / 说明                                         |
| ------------------------- | ---------------------------------------------------------- |
| `--data_dir`              | Directory with stock data / 数据目录                       |
| `--device`                | Device for computation (`cuda` or `cpu`) / 训练设备        |
| `--epochs`                | NN training epochs / 神经网络训练轮数                      |
| `--learning_rate`         | NN learning rate / 神经网络学习率                          |
| `--train_period_days`     | Days for training period / 训练周期天数                    |
| `--predict_period_days`   | Days for prediction period / 预测周期天数                  |
| `--gap_days`              | Gap between training and prediction / 训练与预测间隔天数   |
| `--train_batch_size`      | NN training batch size / 神经网络训练批次大小              |
| `--predict_batch_size`    | NN prediction batch size / 神经网络预测批次大小            |
| `--model_save_dir`        | Directory to save models / 模型保存目录                    |
| `--predictions_save_dir`  | Directory to save predictions / 预测结果保存目录           |
| `--project_name`          | Project/experiment name / 项目名称                         |
| `--model_save_frequency`  | Save NN model every N epochs / 每 N 轮保存神经网络模型     |
| `--slide_period_days`     | Sliding window length / 滑动窗口长度                       |
| `--filter_file_path`      | Feature filter file path / 特征筛选文件路径                |
| `--num_periods`           | Number of periods to process / 处理周期数                  |
| `--lgb_n_estimators`      | LightGBM boosting iterations / LightGBM 迭代轮数           |
| `--early_stopping_rounds` | LightGBM early stopping rounds / LightGBM 早停轮数         |
| `--valid_size`            | LightGBM validation set proportion / LightGBM 验证集比例   |
| `--verbose_eval`          | LightGBM evaluation log frequency / LightGBM 日志输出间隔  |
| `--lgb_params`            | LightGBM hyperparameters JSON string / LightGBM 超参数配置 |

Use `python src/main.py --help` to see all options / 查看所有参数：

```bash
python src/main.py --help
```

---

## ▶️ Usage / 使用方法

Run the pipeline for Neural Network (NN) / 执行神经网络预测：

```bash
python src/main_neural_network.py \
    --data_dir /home/user0/data/StockDailyData/ \
    --device cuda \
    --train_period_days 720 \
    --predict_period_days 60 \
    --gap_days 20
```

Run the pipeline for LightGBM / Ensemble / 执行 LightGBM / 集成模型预测：

```bash
python src/main_ensemble.py \
    --data_dir /home/user0/data/StockDailyData/ \
    --train_period_days 720 \
    --predict_period_days 60 \
    --gap_days 20
```

### Pipeline steps / 流程

**Neural Network** (NN) / 神经网络：

1. Load and preprocess stock data / 加载并预处理股票数据
2. Generate training and prediction periods / 生成训练与预测周期
3. Train Neural Network / 训练神经网络
4. Save NN models per epoch / 每轮保存神经网络模型
5. Make predictions using trained NN / 使用训练好的神经网络进行预测
6. Log progress & metrics / 日志记录训练与预测信息

**LightGBM / Ensemble** / LightGBM / 集成模型：

1. Load and preprocess stock data / 加载并预处理股票数据
2. Generate training and prediction periods / 生成训练与预测周期
3. Train LightGBM / 训练 LightGBM 模型
4. Save trained LightGBM model / 保存训练好的 LightGBM 模型
5. Make predictions using trained LightGBM / 使用训练好的 LightGBM 模型进行预测
6. Log progress & metrics / 日志记录训练与预测信息

---

## 📖 Documentation / 文档

- `docs/flowchart.md` contains pipeline workflow diagrams / 流程图
- Additional documentation for NN & LightGBM:

  - Data preprocessing / 数据预处理
  - Feature selection / 特征选择
  - Model architecture & hyperparameters / 模型结构与超参数

---

## 📝 Logging / 日志

Logs are saved in `log_dir` / 日志保存在 `log_dir` 目录：

- Start & end of each period / 每个周期的开始与结束
- NN loss per epoch / 神经网络每轮损失
- LightGBM metrics / LightGBM 评估指标
- Prediction start & end / 预测开始与结束
- Errors / 错误信息

---

## 🌟 Contributions / 欢迎贡献

You can contribute / 贡献建议：

- New NN architectures / 新的神经网络模型
- Feature engineering methods / 新的特征工程方法
- Pipeline optimization / 流水线优化与加速
- LightGBM enhancements / LightGBM 优化
