import numpy as np
import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):
    def __init__(self, data_list):
        """
        data_list: list of (data_df, target)
        """
        self.samples = []
        self.max_stocks = 0

        for data, target in data_list:
            stock_codes_data = data.iloc[:, 1].astype(str).str.zfill(6)
            stock_codes_data = stock_codes_data.astype(np.int64).values
            stock_codes_target = target.index.astype(str).str.zfill(6)
            stock_codes_target = stock_codes_target.astype(np.int64)
            mask = np.isin(stock_codes_data, stock_codes_target.values)
            common_codes = stock_codes_data[mask]
            dates = data.iloc[:, 0].values.astype(np.int64)[mask]
            features = data.iloc[:, 2:].values.astype(np.float32)[mask]
            target_aligned = target.loc[[f"{c:06d}" for c in common_codes]].values
            if target_aligned.ndim == 1:
                labels = target_aligned.astype(np.float32)
            else:
                labels = target_aligned.astype(np.float32)
            n = len(common_codes)
            self.max_stocks = max(self.max_stocks, n)
            self.samples.append((dates, common_codes, features, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dates_np, codes_np, feats_np, labels_np = self.samples[idx]
        n, feat_dim = feats_np.shape
        max_n = self.max_stocks
        valid_n = min(n, max_n)
        features = torch.zeros((max_n, feat_dim), dtype=torch.float32)
        features[:valid_n] = torch.from_numpy(feats_np[:valid_n])
        if labels_np.ndim == 1:
            labels = torch.zeros((max_n,), dtype=torch.float32)
            labels[:valid_n] = torch.from_numpy(labels_np[:valid_n])
        else:
            label_dim = labels_np.shape[1]
            labels = torch.zeros((max_n, label_dim), dtype=torch.float32)
            labels[:valid_n] = torch.from_numpy(labels_np[:valid_n])
        date = torch.zeros((max_n,), dtype=torch.int64)
        date[:valid_n] = torch.from_numpy(dates_np[:valid_n])
        stock_code = torch.zeros((max_n,), dtype=torch.int64)
        stock_code[:valid_n] = torch.from_numpy(codes_np[:valid_n])
        mask = torch.zeros((max_n,), dtype=torch.bool)
        mask[:valid_n] = True
        return date, stock_code, features, labels, mask


class StockDatasetWithoutMask(Dataset):
    def __init__(self, data_list):
        self.samples = []
        for data, target in data_list:
            dates = data.iloc[:, 0].values.astype(np.int64)
            codes = data.iloc[:, 1].astype(str).str.zfill(6).astype(np.int64).values
            features = data.iloc[:, 2:].values.astype(np.float32)
            if target is None:
                self.samples.append((dates, codes, features, None))
                continue
            target_codes = target.index.astype(str).str.zfill(6).astype(np.int64)
            mask = np.isin(codes, target_codes.values)
            dates = dates[mask]
            codes = codes[mask]
            features = features[mask]
            labels_np = target.loc[[f"{c:06d}" for c in codes]].values
            labels = labels_np.astype(np.float32)
            self.samples.append((dates, codes, features, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dates, codes, features, labels = self.samples[idx]
        date = torch.from_numpy(dates)
        stock_code = torch.from_numpy(codes)
        features = torch.from_numpy(features)
        if labels is None:
            return date, stock_code, features, None
        return date, stock_code, features, torch.from_numpy(labels)


def get_dataloader(data_list, batch_size: int = 64, shuffle: bool = False) -> Dataset:
    """
    Function to generate a DataLoader from a list of (data, target) tuples with mask.
    """
    dataset = StockDataset(data_list)
    return dataset, None


def get_dataloader_predict(data_list, batch_size: int = 64, shuffle: bool = False) -> Dataset:
    """
    Function to generate a DataLoader from a list of (data, target) tuples without mask.
    """
    dataset = StockDatasetWithoutMask(data_list)
    return dataset, None
