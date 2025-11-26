import numpy as np
import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a FEA file into a pandas DataFrame."""
    return pd.read_feather(file_path)


def ensure_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the first column is 'stock_code' (string), and other columns are numeric."""
    df[df.columns[0]] = df[df.columns[0]].astype(str)
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # 'coerce' will turn invalid values into NaN
    return df


def drop_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with any missing values in the DataFrame."""
    return df.dropna()


def fill_missing_values(df: pd.DataFrame, fill_value: float = 0.0) -> pd.DataFrame:
    """Fill missing values in the DataFrame with a specified fill value."""
    return df.fillna(fill_value).replace([np.inf, -np.inf], fill_value)


def normalize_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Normalize specified columns in the DataFrame to the range [0, 1]."""
    for col in columns:
        min_val = df[col].min()
        max_val = df[col].max()
        df[col] = (df[col] - min_val) / (max_val - min_val)
    return df


def standardize_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Standardize specified columns in the DataFrame to have mean 0 and std 1."""
    df[columns] = df[columns].apply(lambda col: (col - col.mean()) / col.std())
    return df


def winsorize_columns(df: pd.DataFrame, columns: list, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> pd.DataFrame:
    """Winsorize specified columns in the DataFrame based on given quantiles."""
    df[columns] = df[columns].apply(lambda col: np.clip(col, col.quantile(lower_quantile), col.quantile(upper_quantile)))
    return df


def iqr_columns(df: pd.DataFrame, columns: list, factor: float = 1.5) -> pd.DataFrame:
    """Remove outliers from specified columns in the DataFrame using the IQR method."""
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df


def log_transform_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Apply log transformation to specified columns in the DataFrame."""
    df[columns] = df[columns].apply(lambda col: np.log1p(col))
    return df
