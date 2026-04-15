import os
import pandas as pd
from tqdm import tqdm

# 原始标签（n日）
for n in range(1, 21):
    factor_dir = "/home/haris/raid0/shared/haris/mydata_20260127/concat_daily_factor"
    save_dir = f"/home/haris/raid0/shared/haris/mydata_20260127/concat_daily_factor_with_label_{n}"
    os.makedirs(save_dir, exist_ok=True)
    label_df = pd.read_feather(f"/home/haris/raid0/shared/haris/mydata_20260127/label_{n}.fea")
    factor_files = sorted(os.listdir(factor_dir))[:-n - 1]
    for factor_file in tqdm(factor_files[-1:]):  # 只计算最后一天
        date = factor_file[:8]
        factor = pd.read_feather(os.path.join(factor_dir, factor_file))
        label = label_df.loc[date].to_frame(name="label")
        label_aligned = label.reindex(factor.index)
        factor_with_label = pd.concat([factor, label_aligned], axis=1)
        factor_with_label.to_feather(os.path.join(save_dir, factor_file))

print("Done")
