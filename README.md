# 🚀 StockPredictor

## 📊 Overview

**Predictor** is a Python-based stock prediction pipeline designed to train models on historical trading data and generate predictions for future stock prices. It integrates data preprocessing, feature selection, model training, and prediction modules with logging and configurable parameters. This project is structured to facilitate research, experimentation, and deployment of quantitative trading models.

---

## 📁 Directory Structure

```
predictor/
├── README.md                # Project overview and usage instructions 📝
├── docs/                    # Documentation 📖
│   └── flowchart.md         # Workflow flowchart 🗺️
├── scripts/                 # Auxiliary scripts for automation ⚡
└── src/                     # Source code 💻
    ├── main.py              # Main entry point ▶️
    ├── pipeline/            # Core pipeline modules 🔄
    │   ├── data.py          # Data loading, preprocessing 🧹
    │   ├── filter.py        # Feature selection and filtering 🔎
    │   ├── predict.py       # Model inference and prediction saving 📝
    │   └── train.py         # Model training utilities 🏋️
    └── utils/               # Utility functions 🛠️
        ├── function.py      # General helper functions 🔧
        ├── logger.py        # Logger setup 📝
        └── model.py         # Model definitions and dataloader utilities 🧠
```

---

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/predictor.git
cd predictor
```

2. (Optional) Create a Python virtual environment:

```bash
python -m venv myenv
source myenv/bin/activate   # Linux/macOS
myenv\Scripts\activate      # Windows
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

**Typical dependencies** include:

* `torch` 🧠
* `pandas` 🐼
* `numpy` 🔢
* `tqdm` ⏳

---

## ⚙️ Configuration

All configurable parameters are defined via command-line arguments. Example parameters:

| Argument                 | Description                                            |
| ------------------------ | ------------------------------------------------------ |
| `--data_dir`             | Directory containing stock data files 📂               |
| `--device`               | Device to use for computation (`cuda` or `cpu`) 💻     |
| `--epochs`               | Number of training epochs (default: 10) ⏱️             |
| `--filter_file_path`     | Path to filter index file 📝                           |
| `--gap_days`             | Days between end of training and start of prediction ⏳ |
| `--learning_rate`        | Learning rate for training (default: 0.001) ⚡          |
| `--log_dir`              | Directory to save logs 🗂️                             |
| `--model_save_dir`       | Directory to save models 💾                            |
| `--num_periods`          | Number of periods to process (default: all) 🔢         |
| `--predict_batch_size`   | Batch size for prediction 📦                           |
| `--predict_period_days`  | Number of days for prediction period 📅                |
| `--predictions_save_dir` | Directory to save predictions 📝                       |
| `--project_name`         | Name of the project/experiment 🏷️                     |
| `--model_save_frequency` | Frequency (in epochs) to save the model 💾             |
| `--slide_period_days`    | Sliding window length in days 🔄                       |
| `--train_batch_size`     | Batch size for training 📦                             |
| `--train_period_days`    | Number of days for training period 📅                  |

You can see all options by running:

```bash
python src/main.py --help
```

---

## ▶️ Usage

Run the full prediction pipeline, e.g.:

```bash
python src/main.py \
    --data_dir /home/user0/data/StockDailyData/ \
    --device cuda \
    --train_period_days 720 \
    --predict_period_days 60 \
    --gap_days 20
```

The pipeline will:

1. Load and preprocess the stock data 🧹.
2. Generate training and prediction periods 📅.
3. Train models for each period 🏋️.
4. Save trained models and predictions 💾.
5. Log progress and metrics 📝.

---

## 📖 Documentation

* `docs/flowchart.md` contains the pipeline workflow diagram 🗺️.
* Additional documentation can be added for:

  * Data preprocessing rules 🧹
  * Feature selection strategy 🔎
  * Model architecture and hyperparameters 🧠

---

## 📝 Logging

All logs are saved in the configured `log_dir`. The logs include:

* Start and end of each training period ⏱️
* Loss per epoch 📉
* Prediction start and end 🔮
* Errors or warnings ⚠️

---

## 🌟 Welcome Contributions

Feel free to fork the repository and submit pull requests. Suggested contributions:

* New models or architectures 🏗️
* Additional feature engineering methods ✨
* Pipeline optimization and speed improvements ⚡
