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
        stock_codes_data = data.iloc[:, 1].astype(str).str.zfill(6)
        stock_codes_target = target.index.astype(str).str.zfill(6)
        common_codes = stock_codes_data[stock_codes_data.isin(stock_codes_target)]
        mask = stock_codes_data.isin(common_codes)
        data_filtered = data.loc[mask].copy()
        data_filtered.iloc[:, 1] = data_filtered.iloc[:, 1].astype(str).str.zfill(6)
        target_filtered = target.loc[common_codes].copy()
        date = torch.tensor(data_filtered.iloc[:, 0].astype(int).values, dtype=torch.int64)
        stock_code = torch.tensor(data_filtered.iloc[:, 1].astype(int).values, dtype=torch.int64)
        features = torch.tensor(data_filtered.iloc[:, 2:].values, dtype=torch.float32)
        label = torch.tensor(target_filtered.values, dtype=torch.float32)
        return date, stock_code, features, label

    # def __getitem__(self, idx):
    #     data, target = self.data_list[idx]
    #     stock_codes_data = data.iloc[:, 0].astype(str).str.zfill(6)
    #     stock_codes_target = target.index.astype(str).str.zfill(6)
    #     common_codes = stock_codes_data[stock_codes_data.isin(stock_codes_target)]
    #     mask = stock_codes_data.isin(common_codes)
    #     data_filtered = data.loc[mask].copy()
    #     data_filtered.iloc[:, 0] = data_filtered.iloc[:, 0].astype(str).str.zfill(6)
    #     target_filtered = target.loc[common_codes].copy()
    #     date = torch.tensor(data_filtered.iloc[:, 1].astype(int).values, dtype=torch.int64)
    #     stock_code = torch.tensor(data_filtered.iloc[:, 0].astype(int).values, dtype=torch.int64)
    #     features = torch.tensor(data_filtered.iloc[:, 2:].values, dtype=torch.float32)
    #     label = torch.tensor(target_filtered.values, dtype=torch.float32)
    #     return date, stock_code, features, label


def get_dataloader(data_list, batch_size: int = 64, shuffle: bool = False) -> DataLoader:
    """
    Function to generate a DataLoader from a list of (data, target) tuples.
    """
    dataset = StockDataset(data_list)
    return dataset, DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
