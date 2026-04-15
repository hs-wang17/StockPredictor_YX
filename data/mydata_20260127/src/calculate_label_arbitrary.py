import os
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

MIN_DATA_DIR = "/home/haris/data/min_data/"
MIN_DATA_DIR_LIST = sorted(os.listdir(MIN_DATA_DIR))
INDEX_WEIGHT_DIR = "/home/haris/data/IndexWeightData"

all_stk_close_1300_list = []
for file in tqdm(MIN_DATA_DIR_LIST):
    date = file.replace(".fea", "")
    stk_df = pd.read_feather(os.path.join(MIN_DATA_DIR, file))
    all_stk_close_1300 = stk_df[stk_df["time"] == 1300][["code", "close"]].set_index("code").rename(columns={"close": date})
    all_stk_close_1300_list.append(all_stk_close_1300)
all_stk_close_1300_df = pd.concat(all_stk_close_1300_list, axis=1).T
all_stk_close_1300_df.index.name = "date"
adjfactor = pd.read_feather("/home/haris/data/data_frames/stk_adjfactor.feather")
adjfactor = adjfactor.reindex(index=all_stk_close_1300_df.index, columns=all_stk_close_1300_df.columns)
all_stk_close_1300_df = all_stk_close_1300_df * adjfactor

for n in range(1, 21):
    stk_ret_n = (all_stk_close_1300_df.pct_change(n).shift(-n)).dropna(how="all")
    idx_ret_n_list, dates_list = [], []
    for file in tqdm(MIN_DATA_DIR_LIST[:-n]):
        date = file.replace(".fea", "")
        stk_ret = stk_ret_n.loc[date].dropna()
        idx_weight_df = pd.read_feather(os.path.join(INDEX_WEIGHT_DIR, file))
        idx_weight = idx_weight_df[idx_weight_df["index_name"] == "ZZ1000"].set_index("stock_code")["stock_weight"]
        common_codes = stk_ret.index.intersection(idx_weight.index)
        stk_ret = stk_ret.loc[common_codes]
        idx_weight = idx_weight.loc[common_codes]
        idx_ret = (stk_ret * idx_weight).sum() / idx_weight.sum()
        idx_ret_n_list.append(idx_ret)
        dates_list.append(date)
    idx_ret_n = pd.Series(idx_ret_n_list, index=dates_list)
    label_n = stk_ret_n.sub(idx_ret_n, axis=0).dropna(how="all")

    high_limit = pd.read_feather("/home/haris/data/data_frames/stk_ztprice.feather").replace(0, np.nan).ffill()
    open = pd.read_feather("/home/haris/data/data_frames/stk_open.feather").replace(0, np.nan).ffill()
    zt_df = (open == high_limit).fillna(False).astype(int)
    st_df = pd.read_feather("/home/haris/data/data_frames/stk_is_st_stock.feather").fillna(False).astype(int)
    stop_df = pd.read_feather("/home/haris/data/data_frames/stk_is_stop_stock.feather").fillna(False).astype(int)
    zt_st_stop_df = zt_df | st_df | stop_df
    mask = zt_st_stop_df.reindex(index=label_n.index, columns=label_n.columns)
    label_n = label_n.mask(mask == 1).dropna(how="all")
    # label_n.to_feather("/home/haris/project/backtester/data/label_{n}.fea")
    label_n.to_feather(f"/home/haris/raid0/shared/haris/mydata_20260127/label_{n}.fea")
    trade_date_df = pd.DataFrame({"trade_date": label_n.index})
    trade_date_df.reset_index(drop=True).to_feather(f"/home/haris/raid0/shared/haris/mydata_20260127/trade_date_{n}.fea")

print("Done")
