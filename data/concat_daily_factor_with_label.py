import os
import pandas as pd
from tqdm import tqdm

factor_dir = "/home/haris/mydata/concat_daily_factor"
save_dir = "/home/haris/mydata/concat_daily_factor_with_label"
os.makedirs(save_dir, exist_ok=True)
label_df = pd.read_feather("/home/haris/mydata/label.fea")

factor_files = sorted(os.listdir(factor_dir))
for factor_file in tqdm(factor_files):
    date = factor_file[:8]
    factor = pd.read_feather(os.path.join(factor_dir, factor_file))
    label = label_df.loc[date].dropna().to_frame(name="label")
    label.columns = ["label"]
    common_index = factor.index.intersection(label.index)
    factor_aligned = factor.loc[common_index]
    label_aligned = label.loc[common_index]
    factor_with_label = pd.concat([factor_aligned, label_aligned], axis=1)
    factor_with_label.to_feather(os.path.join(save_dir, factor_file))

print("Done")
