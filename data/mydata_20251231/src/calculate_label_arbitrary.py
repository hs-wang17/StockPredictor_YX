import numpy as np
import pandas as pd

vwap = pd.read_feather("/home/haris/data/trade_support_data/vwap.fea")
adjfactor = pd.read_feather("/home/haris/data/data_frames/stk_adjfactor.feather")
adjfactor = adjfactor.reindex(index=vwap.index, columns=vwap.columns)
vwap = vwap * adjfactor

# 原始标签（n日）
for n in range(1, 21):
    idx_open = pd.read_feather("/home/haris/data/data_frames/idx_open.feather")
    idx_ret_n = idx_open.pct_change(n).shift(-n - 1).dropna(how="all")["中证1000"].squeeze()
    stk_ret_n = vwap.pct_change(n).shift(-n - 1).dropna(how="all")
    label_n = stk_ret_n.sub(idx_ret_n, axis=0).dropna(how="all")

    high_limit = pd.read_feather("/home/haris/data/data_frames/stk_ztprice.feather").replace(0, np.nan).ffill()
    open = pd.read_feather("/home/haris/data/data_frames/stk_open.feather").replace(0, np.nan).ffill()
    zt_df = (open == high_limit).shift(-1).fillna(False).astype(int)
    st_df = pd.read_feather("/home/haris/data/data_frames/stk_is_st_stock.feather").shift(-1).fillna(False).astype(int)
    stop_df = pd.read_feather("/home/haris/data/data_frames/stk_is_stop_stock.feather").shift(-1).fillna(False).astype(int)
    zt_st_stop_df = zt_df | st_df | stop_df
    mask = zt_st_stop_df.reindex(index=label_n.index, columns=label_n.columns)
    label_n = label_n.mask(mask == 1).dropna(how="all")
    # label_n.to_feather("/home/haris/project/backtester/data/label_{n}.fea")
    label_n.to_feather(f"/home/haris/raid0/shared/haris/mydata_20251231/label_{n}.fea")
    trade_date_df = pd.DataFrame({"trade_date": label_n.index})
    trade_date_df.reset_index(drop=True).to_feather(f"/home/haris/raid0/shared/haris/mydata_20251231/trade_date_{n}.fea")

print("Done")
