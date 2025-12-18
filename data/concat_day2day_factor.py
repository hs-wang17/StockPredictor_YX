import numpy as np
import pandas as pd
import os
from tqdm import tqdm

data_dir = "/home/haris/data/DailyFactors/day2day"
save_dir = "/home/haris/mydata/concat_day2day_factor"
factor_types = os.listdir(data_dir)
date_list = []
for factor_type in factor_types:
    factor_type_dir = os.path.join(data_dir, factor_type)
    dates = os.listdir(factor_type_dir)
    date_list.extend(dates)
date_list = sorted(list(set(date_list)))

for date in tqdm(date_list):
    df_list = []
    for factor_type in factor_types:
        factor_path = os.path.join(data_dir, factor_type, date)
        df = pd.read_feather(factor_path).set_index("index")
        df_list.append(df)
    if df_list:
        concat_df = pd.concat(df_list, axis=1)
        concat_df.columns = [f"factor_{str(i).zfill(3)}" for i in np.arange(len(concat_df.columns))]
        save_path = os.path.join(save_dir, f"{date}")
        concat_df.to_feather(save_path)

trade_date_df = pd.DataFrame({"trade_date": [date[:8] for date in date_list[:-21]]})  # Exclude last 21 dates to ensure label availability
trade_date_df.to_feather("/home/haris/mydata/trade_date.fea")

print("Done")
