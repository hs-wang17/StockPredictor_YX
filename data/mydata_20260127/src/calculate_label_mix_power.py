import os
import numpy as np
import pandas as pd
from tqdm import tqdm

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

stk_ret_2 = (all_stk_close_1300_df.pct_change(2).shift(-2)).dropna(how="all")
idx_ret_2_list, dates_list = [], []
for file in tqdm(MIN_DATA_DIR_LIST[:-2]):
    date = file.replace(".fea", "")
    stk_ret = stk_ret_2.loc[date].dropna()
    idx_weight_df = pd.read_feather(os.path.join(INDEX_WEIGHT_DIR, file))
    idx_weight = idx_weight_df[idx_weight_df["index_name"] == "ZZ1000"].set_index("stock_code")["stock_weight"]
    common_codes = stk_ret.index.intersection(idx_weight.index)
    stk_ret = stk_ret.loc[common_codes]
    idx_weight = idx_weight.loc[common_codes]
    idx_ret = (stk_ret * idx_weight).sum() / idx_weight.sum()
    idx_ret_2_list.append(idx_ret)
    dates_list.append(date)
idx_ret_2 = pd.Series(idx_ret_2_list, index=dates_list)
label_2 = stk_ret_2.sub(idx_ret_2, axis=0).dropna(how="all")

stk_ret_4 = (all_stk_close_1300_df.pct_change(4).shift(-4)).dropna(how="all")
idx_ret_4_list, dates_list = [], []
for file in tqdm(MIN_DATA_DIR_LIST[:-4]):
    date = file.replace(".fea", "")
    stk_ret = stk_ret_4.loc[date].dropna()
    idx_weight_df = pd.read_feather(os.path.join(INDEX_WEIGHT_DIR, file))
    idx_weight = idx_weight_df[idx_weight_df["index_name"] == "ZZ1000"].set_index("stock_code")["stock_weight"]
    common_codes = stk_ret.index.intersection(idx_weight.index)
    stk_ret = stk_ret.loc[common_codes]
    idx_weight = idx_weight.loc[common_codes]
    idx_ret = (stk_ret * idx_weight).sum() / idx_weight.sum()
    idx_ret_4_list.append(idx_ret)
    dates_list.append(date)
idx_ret_4 = pd.Series(idx_ret_4_list, index=dates_list)
label_4 = stk_ret_4.sub(idx_ret_4, axis=0).dropna(how="all")

stk_ret_8 = (all_stk_close_1300_df.pct_change(8).shift(-8)).dropna(how="all")
idx_ret_8_list, dates_list = [], []
for file in tqdm(MIN_DATA_DIR_LIST[:-8]):
    date = file.replace(".fea", "")
    stk_ret = stk_ret_8.loc[date].dropna()
    idx_weight_df = pd.read_feather(os.path.join(INDEX_WEIGHT_DIR, file))
    idx_weight = idx_weight_df[idx_weight_df["index_name"] == "ZZ1000"].set_index("stock_code")["stock_weight"]
    common_codes = stk_ret.index.intersection(idx_weight.index)
    stk_ret = stk_ret.loc[common_codes]
    idx_weight = idx_weight.loc[common_codes]
    idx_ret = (stk_ret * idx_weight).sum() / idx_weight.sum()
    idx_ret_8_list.append(idx_ret)
    dates_list.append(date)
idx_ret_8 = pd.Series(idx_ret_8_list, index=dates_list)
label_8 = stk_ret_8.sub(idx_ret_8, axis=0).dropna(how="all")

stk_ret_16 = (all_stk_close_1300_df.pct_change(16).shift(-16)).dropna(how="all")
idx_ret_16_list, dates_list = [], []
for file in tqdm(MIN_DATA_DIR_LIST[:-16]):
    date = file.replace(".fea", "")
    stk_ret = stk_ret_16.loc[date].dropna()
    idx_weight_df = pd.read_feather(os.path.join(INDEX_WEIGHT_DIR, file))
    idx_weight = idx_weight_df[idx_weight_df["index_name"] == "ZZ1000"].set_index("stock_code")["stock_weight"]
    common_codes = stk_ret.index.intersection(idx_weight.index)
    stk_ret = stk_ret.loc[common_codes]
    idx_weight = idx_weight.loc[common_codes]
    idx_ret = (stk_ret * idx_weight).sum() / idx_weight.sum()
    idx_ret_16_list.append(idx_ret)
    dates_list.append(date)
idx_ret_16 = pd.Series(idx_ret_16_list, index=dates_list)
label_16 = stk_ret_16.sub(idx_ret_16, axis=0).dropna(how="all")

label_mix = (label_2 * 0.1 + label_4 * 0.2 + label_8 * 0.4 + label_16 * 0.8) / 1.5
high_limit = pd.read_feather("/home/haris/data/data_frames/stk_ztprice.feather").replace(0, np.nan).ffill()
open = pd.read_feather("/home/haris/data/data_frames/stk_open.feather").replace(0, np.nan).ffill()
zt_df = (open == high_limit).fillna(False).astype(int)
st_df = pd.read_feather("/home/haris/data/data_frames/stk_is_st_stock.feather").fillna(False).astype(int)
stop_df = pd.read_feather("/home/haris/data/data_frames/stk_is_stop_stock.feather").fillna(False).astype(int)
zt_st_stop_df = zt_df | st_df | stop_df
mask = zt_st_stop_df.reindex(index=label_mix.index, columns=label_mix.columns)
label_mix = label_mix.mask(mask == 1).dropna(how="all")
label_mix.to_feather("/home/haris/project/backtester/data/label_mix_power.fea")
label_mix.to_feather("/home/haris/raid0/shared/haris/mydata_20260127/label_mix_power.fea")
trade_date_df = pd.DataFrame({"trade_date": label_mix.index})
trade_date_df.reset_index(drop=True).to_feather("/home/haris/raid0/shared/haris/mydata_20260127/trade_date_mix_power.fea")

print("Done")
