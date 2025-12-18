import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

def read_filter_index(file_path: str, period_index: int) -> list:
    """Read filter index from a FEA file."""
    if os.path.exists(file_path):
        return pd.read_feather(file_path)[period_index].tolist()
    else:
        return None

def calculate_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the Pearson correlation matrix for all features."""
    correlation_matrix = df.corr()
    return correlation_matrix

def filter_highly_correlated_features(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """Remove features that are highly correlated (correlation > threshold)."""
    correlation_matrix = calculate_correlation(df)
    columns_to_drop = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                colname = correlation_matrix.columns[i]
                columns_to_drop.add(colname)
    df_filtered = df.drop(columns=columns_to_drop)
    return df_filtered

def calculate_ic(df: pd.DataFrame, label: str) -> pd.Series:
    """Calculate Information Coefficient (IC) between features and the label."""
    ic_values = {}
    for col in df.columns:
        if col != label:
            ic_values[col] = pearsonr(df[col], df[label])[0]  # Pearson correlation coefficient
    ic_series = pd.Series(ic_values)
    return ic_series

def calculate_ir(ic_series: pd.Series) -> float:
    """Calculate Information Ratio (IR) based on IC values."""
    return ic_series.mean() / ic_series.std() if ic_series.std() != 0 else 0

def select_features_by_ic_ir(df: pd.DataFrame, label: str, ic_threshold: float = 0.1, ir_threshold: float = 0.2) -> pd.DataFrame:
    """Select features based on their IC and IR values."""
    # Calculate IC for all features
    ic_values = calculate_ic(df, label)

    # Filter features based on IC threshold
    selected_features = ic_values[ic_values > ic_threshold].index
    
    # Calculate IR for the selected features
    selected_df = df[selected_features]
    ic_series = calculate_ic(selected_df, label)
    ir_value = calculate_ir(ic_series)
    
    if ir_value < ir_threshold:
        print(f"Warning: IR value is low ({ir_value}), consider revisiting feature selection.")
    
    return selected_df

def recursive_feature_elimination(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Use RFE to select the best features based on a linear model."""
    X = df.drop(columns=[label])
    y = df[label]
    
    model = LinearRegression()
    rfe = RFE(model, n_features_to_select=10)  # Select top 10 features
    X_rfe = rfe.fit_transform(X, y)
    
    selected_features = X.columns[rfe.support_]
    return df[selected_features]

# # Example usage:
# # Assuming you have a DataFrame `df` where 'stock_code' is the first column and 'label' is the target variable

# # 1. Remove highly correlated features
# df_filtered = filter_highly_correlated_features(df.iloc[:, 1:])  # exclude stock_code column
# print("Filtered DataFrame based on correlation")

# # 2. Select features based on IC & IR
# df_ic_ir_selected = select_features_by_ic_ir(df_filtered, label='label', ic_threshold=0.1, ir_threshold=0.2)
# print("Selected features based on IC and IR")

# # 3. Use RFE for feature selection
# df_rfe_selected = recursive_feature_elimination(df_filtered, label='label')
# print("Selected features using RFE")
