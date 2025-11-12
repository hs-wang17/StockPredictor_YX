import torch
from torch.utils.data import DataLoader, Dataset

class StockDataset(Dataset):
    def __init__(self, data_list):
        """
        Args:
            data_list (list): A list of tuples where each tuple is (data, target)
            feature_columns (list): List of columns to use as features
            label_column (str): The column to use as the target (label)
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data, target = self.data_list[idx]
        date = torch.tensor(data.iloc[:, 0].astype(int).values, dtype=torch.int64)
        stock_code = torch.tensor(data.iloc[:, 1].astype(int).values, dtype=torch.int64)
        features = torch.tensor(data.iloc[:, 2:].values, dtype=torch.float32)
        label = torch.tensor(target.values, dtype=torch.float32)
        return date, stock_code, features, label

def get_dataloader(data_list, batch_size: int = 64, shuffle: bool = False) -> DataLoader:
    """
    Function to generate a DataLoader from a list of (data, target) tuples.
    """
    dataset = StockDataset(data_list)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
