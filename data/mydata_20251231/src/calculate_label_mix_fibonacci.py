import numpy as np
import pandas as pd

vwap = pd.read_feather("/home/haris/data/trade_support_data/vwap.fea")
adjfactor = pd.read_feather("/home/haris/data/data_frames/stk_adjfactor.feather")
adjfactor = adjfactor.reindex(index=vwap.index, columns=vwap.columns)
vwap = vwap * adjfactor

# 原始标签（2/3/5/8/13/21日）
idx_open = pd.read_feather("/home/haris/data/data_frames/idx_open.feather")
idx_ret_2 = idx_open.pct_change(2).shift(-3).dropna(how="all")["中证1000"].squeeze()
idx_ret_3 = idx_open.pct_change(3).shift(-4).dropna(how="all")["中证1000"].squeeze()
idx_ret_5 = idx_open.pct_change(5).shift(-6).dropna(how="all")["中证1000"].squeeze()
idx_ret_8 = idx_open.pct_change(8).shift(-9).dropna(how="all")["中证1000"].squeeze()
idx_ret_13 = idx_open.pct_change(13).shift(-14).dropna(how="all")["中证1000"].squeeze()
idx_ret_21 = idx_open.pct_change(21).shift(-22).dropna(how="all")["中证1000"].squeeze()

stk_ret_2 = vwap.pct_change(2).shift(-3).dropna(how="all")
stk_ret_3 = vwap.pct_change(3).shift(-4).dropna(how="all")
stk_ret_5 = vwap.pct_change(5).shift(-6).dropna(how="all")
stk_ret_8 = vwap.pct_change(8).shift(-9).dropna(how="all")
stk_ret_13 = vwap.pct_change(13).shift(-14).dropna(how="all")
stk_ret_21 = vwap.pct_change(21).shift(-22).dropna(how="all")

label_2 = stk_ret_2.sub(idx_ret_2, axis=0).dropna(how="all")
label_3 = stk_ret_3.sub(idx_ret_3, axis=0).dropna(how="all")
label_5 = stk_ret_5.sub(idx_ret_5, axis=0).dropna(how="all")
label_8 = stk_ret_8.sub(idx_ret_8, axis=0).dropna(how="all")
label_13 = stk_ret_13.sub(idx_ret_13, axis=0).dropna(how="all")
label_21 = stk_ret_21.sub(idx_ret_21, axis=0).dropna(how="all")
label_mix = (label_2 * 0.1 + label_3 * 0.2 + label_5 * 0.4 + label_8 * 0.8 + label_13 * 1.6 + label_21 * 3.2) / 6.3

high_limit = pd.read_feather("/home/haris/data/data_frames/stk_ztprice.feather").replace(0, np.nan).ffill()
open = pd.read_feather("/home/haris/data/data_frames/stk_open.feather").replace(0, np.nan).ffill()
zt_df = (open == high_limit).shift(-1).fillna(False).astype(int)
st_df = pd.read_feather("/home/haris/data/data_frames/stk_is_st_stock.feather").shift(-1).fillna(False).astype(int)
stop_df = pd.read_feather("/home/haris/data/data_frames/stk_is_stop_stock.feather").shift(-1).fillna(False).astype(int)
zt_st_stop_df = zt_df | st_df | stop_df
mask = zt_st_stop_df.reindex(index=label_mix.index, columns=label_mix.columns)
label_mix = label_mix.mask(mask == 1).dropna(how="all")
# label_mix.to_feather("/home/haris/project/backtester/data/label_mix_fibonacci.fea")
label_mix.to_feather("/home/haris/raid0/shared/haris/mydata_20251231/label_mix_fibonacci.fea")
trade_date_df = pd.DataFrame({"trade_date": label_mix.index})
trade_date_df.reset_index(drop=True).to_feather("/home/haris/raid0/shared/haris/mydata_20251231/trade_date_mix_fibonacci.fea")

print("Done")
