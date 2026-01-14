import numpy as np
import torch
from torch.utils.data import Dataset


# class StockDataset(Dataset):
#     def __init__(self, data_list):
#         """
#         Args:
#             data_list: list of (data_df, target_series)
#             max_stocks: 股票截面上限
#         """
#         self.data_list = data_list
#         self.max_stocks = get_optimal_max_stocks(data_list)

#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, idx):
#         data, target = self.data_list[idx]
#         stock_codes_data = data.iloc[:, 1].astype(str).str.zfill(6)
#         stock_codes_target = target.index.astype(str).str.zfill(6)
#         common_codes = stock_codes_data[stock_codes_data.isin(stock_codes_target)]
#         mask_df = stock_codes_data.isin(common_codes)
#         data_filtered = data.loc[mask_df]
#         target_filtered = target.loc[common_codes]
#         dates_np = data_filtered.iloc[:, 0].astype(np.int64).values
#         stock_codes_np = data_filtered.iloc[:, 1].astype(np.int64).values
#         features_np = data_filtered.iloc[:, 2:].values.astype(np.float32)
#         labels_np = target_filtered.values.astype(np.float32)
#         n, feature_dim = features_np.shape
#         max_n = self.max_stocks
#         valid_n = min(n, max_n)
#         features = torch.zeros((max_n, feature_dim), dtype=torch.float32)
#         labels = torch.zeros((max_n,), dtype=torch.float32)
#         stock_code = torch.zeros((max_n,), dtype=torch.int64)
#         date = torch.zeros((max_n,), dtype=torch.int64)
#         mask = torch.zeros((max_n,), dtype=torch.bool)
#         features[:valid_n] = torch.from_numpy(features_np[:valid_n])
#         labels[:valid_n] = torch.from_numpy(labels_np[:valid_n])
#         stock_code[:valid_n] = torch.from_numpy(stock_codes_np[:valid_n])
#         date[:valid_n] = torch.from_numpy(dates_np[:valid_n])
#         mask[:valid_n] = True
#         return date, stock_code, features, labels, mask


class StockDataset(Dataset):
    def __init__(self, data_list):
        """
        Args:
            data_list: list of (data_df, target)
                data_df: columns = [date, stock_code, feat1, feat2, ...]
                target:
                    - Series: index=stock_code, values=scalar
                    - DataFrame: index=stock_code, values=(K horizons)
        """
        self.data_list = data_list
        self.max_stocks = self._get_optimal_max_stocks(data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data, target = self.data_list[idx]

        # -------- 股票代码对齐 --------
        stock_codes_data = data.iloc[:, 1].astype(str).str.zfill(6)
        stock_codes_target = target.index.astype(str).str.zfill(6)

        common_codes = stock_codes_data[stock_codes_data.isin(stock_codes_target)]
        mask_df = stock_codes_data.isin(common_codes)

        data_filtered = data.loc[mask_df]
        target_filtered = target.loc[common_codes]

        # -------- numpy 化 --------
        dates_np = data_filtered.iloc[:, 0].astype(np.int64).values
        stock_codes_np = data_filtered.iloc[:, 1].astype(np.int64).values
        features_np = data_filtered.iloc[:, 2:].values.astype(np.float32)

        values = target_filtered.values
        if isinstance(values[0], (list, tuple, np.ndarray)):
            labels_np = np.stack(values).astype(np.float32)
        else:
            labels_np = values.astype(np.float32)

        n, feature_dim = features_np.shape
        max_n = self.max_stocks
        valid_n = min(n, max_n)

        # -------- feature padding --------
        features = torch.zeros((max_n, feature_dim), dtype=torch.float32)
        features[:valid_n] = torch.from_numpy(features_np[:valid_n])

        # -------- label padding --------
        if labels_np.ndim == 1:
            # 标量标签
            labels = torch.zeros((max_n,), dtype=torch.float32)
            labels[:valid_n] = torch.from_numpy(labels_np[:valid_n])
        elif labels_np.ndim == 2:
            # 多期限标签
            label_dim = labels_np.shape[1]
            labels = torch.zeros((max_n, label_dim), dtype=torch.float32)
            labels[:valid_n] = torch.from_numpy(labels_np[:valid_n])
        else:
            raise ValueError(f"Unsupported label shape: {labels_np.shape}")

        # -------- 其他字段 --------
        stock_code = torch.zeros((max_n,), dtype=torch.int64)
        stock_code[:valid_n] = torch.from_numpy(stock_codes_np[:valid_n])

        date = torch.zeros((max_n,), dtype=torch.int64)
        date[:valid_n] = torch.from_numpy(dates_np[:valid_n])

        mask = torch.zeros((max_n,), dtype=torch.bool)
        mask[:valid_n] = True

        return date, stock_code, features, labels, mask

    @staticmethod
    def _get_optimal_max_stocks(data_list):
        max_stocks_actual = 0
        for data, target in data_list:
            stock_codes_data = data.iloc[:, 1].astype(str).str.zfill(6)
            stock_codes_target = target.index.astype(str).str.zfill(6)
            common_codes = stock_codes_data[stock_codes_data.isin(stock_codes_target)]
            max_stocks_actual = max(max_stocks_actual, len(common_codes))
        return max_stocks_actual


# class StockDatasetWithoutMask(Dataset):

#     def __init__(self, data_list):
#         """
#         Args:
#             data_list (list): A list of tuples where each tuple is (data, target)
#         """
#         self.data_list = data_list

#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, idx):
#         data, target = self.data_list[idx]
#         stock_codes_data = data.iloc[:, 1].astype(str).str.zfill(6)
#         if target is not None:
#             stock_codes_target = target.index.astype(str).str.zfill(6)
#             common_codes = stock_codes_data[stock_codes_data.isin(stock_codes_target)]
#             mask = stock_codes_data.isin(common_codes)
#             data_filtered = data.loc[mask].copy()
#             data_filtered.iloc[:, 1] = data_filtered.iloc[:, 1].astype(str).str.zfill(6)
#             target_filtered = target.loc[common_codes].copy()
#             date = torch.tensor(data_filtered.iloc[:, 0].astype(int).values, dtype=torch.int64)
#             stock_code = torch.tensor(data_filtered.iloc[:, 1].astype(int).values, dtype=torch.int64)
#             features = torch.tensor(data_filtered.iloc[:, 2:].values, dtype=torch.float32)
#             label = torch.tensor(target_filtered.values, dtype=torch.float32)
#             return date, stock_code, features, label
#         else:
#             data.iloc[:, 1] = data.iloc[:, 1].astype(str).str.zfill(6)
#             date = torch.tensor(data.iloc[:, 0].astype(int).values, dtype=torch.int64)
#             stock_code = torch.tensor(data.iloc[:, 1].astype(int).values, dtype=torch.int64)
#             features = torch.tensor(data.iloc[:, 2:].values, dtype=torch.float32)
#             return date, stock_code, features, None


class StockDatasetWithoutMask(Dataset):
    def __init__(self, data_list):
        """
        Args:
            data_list: list of (data_df, target)
                data_df: columns = [date, stock_code, feat1, feat2, ...]
                target:
                    - None
                    - Series: index=stock_code, values=scalar
                    - DataFrame: index=stock_code, values=(K horizons)
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data, target = self.data_list[idx]

        # -------- 股票代码 --------
        stock_codes_data = data.iloc[:, 1].astype(str).str.zfill(6)

        # -------- 特征（始终返回） --------
        dates_np = data.iloc[:, 0].astype(np.int64).values
        stock_codes_np = data.iloc[:, 1].astype(np.int64).values
        features_np = data.iloc[:, 2:].values.astype(np.float32)

        date = torch.from_numpy(dates_np)
        stock_code = torch.from_numpy(stock_codes_np)
        features = torch.from_numpy(features_np)

        # -------- 无 label 情况（预测） --------
        if target is None:
            return date, stock_code, features, None

        # -------- label 对齐 --------
        stock_codes_target = target.index.astype(str).str.zfill(6)
        common_codes = stock_codes_data[stock_codes_data.isin(stock_codes_target)]
        mask = stock_codes_data.isin(common_codes)

        data_filtered = data.loc[mask]
        target_filtered = target.loc[common_codes]

        # -------- 特征对齐后重建 --------
        dates_np = data_filtered.iloc[:, 0].astype(np.int64).values
        stock_codes_np = data_filtered.iloc[:, 1].astype(np.int64).values
        features_np = data_filtered.iloc[:, 2:].values.astype(np.float32)

        date = torch.from_numpy(dates_np)
        stock_code = torch.from_numpy(stock_codes_np)
        features = torch.from_numpy(features_np)

        # -------- label 自适应（核心） --------
        values = target_filtered.values

        if isinstance(values[0], (list, tuple, np.ndarray)):
            # 多期限 label: (N, K)
            labels_np = np.stack(values).astype(np.float32)
        else:
            # 标量 label: (N,)
            labels_np = values.astype(np.float32)

        labels = torch.from_numpy(labels_np)

        return date, stock_code, features, labels


def get_optimal_max_stocks(data_list):
    """
    Calculate the optimal max_stocks value based on the data_list.
    """
    max_stocks_actual = 0
    for data, target in data_list:
        stock_codes_data = data.iloc[:, 1].astype(str).str.zfill(6)
        stock_codes_target = target.index.astype(str).str.zfill(6)
        common_codes = stock_codes_data[stock_codes_data.isin(stock_codes_target)]
        max_stocks_actual = max(max_stocks_actual, len(common_codes))
    return max_stocks_actual


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
