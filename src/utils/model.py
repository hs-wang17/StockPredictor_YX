import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

# 假设数据已经加载到df中，数据中有"date"、"stock_code"、"features"和"label"
class StockDataset(Dataset):
    def __init__(self, data: pd.DataFrame, feature_columns: list, label_column: str):
        """
        Args:
            data (pd.DataFrame): The dataframe containing stock data
            feature_columns (list): List of columns to use as features
            label_column (str): The column to use as the target (label)
        """
        self.data = data
        self.features = data[feature_columns].values
        self.labels = data[label_column].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# 假设数据已经加载到df中，数据中有"date"、"stock_code"、"features"和"label"
def get_dataloader(df: pd.DataFrame, feature_columns: list, label_column: str, batch_size: int):
    dataset = StockDataset(df, feature_columns, label_column)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)