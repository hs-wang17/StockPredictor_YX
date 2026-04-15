import numpy as np
import pandas as pd

vwap = pd.read_feather("/home/haris/data/trade_support_data/vwap.fea")
adjfactor = pd.read_feather("/home/haris/data/data_frames/stk_adjfactor.feather")
adjfactor = adjfactor.reindex(index=vwap.index, columns=vwap.columns)
vwap = vwap * adjfactor

# 原始标签（2/4/8/16日）
idx_open = pd.read_feather("/home/haris/data/data_frames/idx_open.feather")
idx_ret_2 = idx_open.pct_change(2).shift(-3).dropna(how="all")["中证1000"].squeeze()
idx_ret_4 = idx_open.pct_change(4).shift(-5).dropna(how="all")["中证1000"].squeeze()
idx_ret_8 = idx_open.pct_change(8).shift(-9).dropna(how="all")["中证1000"].squeeze()
idx_ret_16 = idx_open.pct_change(16).shift(-17).dropna(how="all")["中证1000"].squeeze()

stk_ret_2 = vwap.pct_change(2).shift(-3).dropna(how="all")
stk_ret_4 = vwap.pct_change(4).shift(-5).dropna(how="all")
stk_ret_8 = vwap.pct_change(8).shift(-9).dropna(how="all")
stk_ret_16 = vwap.pct_change(16).shift(-17).dropna(how="all")

label_2 = stk_ret_2.sub(idx_ret_2, axis=0).dropna(how="all")
label_4 = stk_ret_4.sub(idx_ret_4, axis=0).dropna(how="all")
label_8 = stk_ret_8.sub(idx_ret_8, axis=0).dropna(how="all")
label_16 = stk_ret_16.sub(idx_ret_16, axis=0).dropna(how="all")
label_mix = (label_2 * 0.1 + label_4 * 0.2 + label_8 * 0.4 + label_16 * 0.8) / 1.5

high_limit = pd.read_feather("/home/haris/data/data_frames/stk_ztprice.feather").replace(0, np.nan).ffill()
open = pd.read_feather("/home/haris/data/data_frames/stk_open.feather").replace(0, np.nan).ffill()
zt_df = (open == high_limit).shift(-1).fillna(False).astype(int)
st_df = pd.read_feather("/home/haris/data/data_frames/stk_is_st_stock.feather").shift(-1).fillna(False).astype(int)
stop_df = pd.read_feather("/home/haris/data/data_frames/stk_is_stop_stock.feather").shift(-1).fillna(False).astype(int)
zt_st_stop_df = zt_df | st_df | stop_df
mask = zt_st_stop_df.reindex(index=label_mix.index, columns=label_mix.columns)
label_mix = label_mix.mask(mask == 1).dropna(how="all")
# label_mix.to_feather("/home/haris/project/backtester/data/label_mix_power.fea")
label_mix.to_feather("/home/haris/raid0/shared/haris/mydata_20251231/label_mix_power.fea")
trade_date_df = pd.DataFrame({"trade_date": label_mix.index})
trade_date_df.reset_index(drop=True).to_feather("/home/haris/raid0/shared/haris/mydata_20251231/trade_date_mix_power.fea")

print("Done")
