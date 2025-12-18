import numpy as np
import pandas as pd

vwap = pd.read_feather("/home/haris/project/backtester/data/vwap.fea")
adjfactor = pd.read_feather("/home/haris/data/data_frames/stk_adjfactor.feather")
adjfactor = adjfactor.reindex(index=vwap.index, columns=vwap.columns)
vwap = vwap * adjfactor

stk_ret = vwap.pct_change(20).shift(-21).dropna(how="all")
idx_open = pd.read_feather("/home/haris/data/data_frames/idx_open.feather")
idx_ret = idx_open.pct_change(20).shift(-21).dropna(how="all")["中证1000"].squeeze()
label = stk_ret.sub(idx_ret, axis=0).dropna(how="all")

high_limit = pd.read_feather("/home/haris/data/data_frames/stk_ztprice.feather").replace(0, np.nan).ffill()
open = pd.read_feather("/home/haris/data/data_frames/stk_open.feather").replace(0, np.nan).ffill()
zt_df = (open == high_limit).shift(-1).fillna(False).astype(int)
st_df = pd.read_feather("/home/haris/data/data_frames/stk_is_st_stock.feather").shift(-1).fillna(False).astype(int)
stop_df = pd.read_feather("/home/haris/data/data_frames/stk_is_stop_stock.feather").shift(-1).fillna(False).astype(int)
zt_st_stop_df = zt_df | st_df | stop_df
mask = zt_st_stop_df.reindex(index=label.index, columns=label.columns)
label = label.mask(mask == 1).dropna(how="all")

label.to_feather("/home/haris/project/backtester/data/label.fea")
label.to_feather("/home/haris/mydata/label.fea")
trade_date_df = pd.DataFrame({"trade_date": label.index})
trade_date_df.reset_index(drop=True).to_feather("/home/haris/mydata/trade_date.fea")

print("Done")
