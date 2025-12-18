from typing import List
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb
import time
import joblib
import multiprocessing as mp
from functools import partial


def _train_single_fold(
    fold: int, train_idx, val_idx, X_train_full, y_train_full, base_model_params, early_stopping_rounds, verbose_eval, save_dir, logger_queue
):
    """在独立进程中训练单折（每折独占一张 GPU）"""
    # 设置当前进程使用的 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(fold)  # fold 0~3 → GPU 0~3

    X_train, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
    y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

    model = lgb.LGBMRegressor(**base_model_params)

    fit_kwargs = {}
    callbacks = []

    fit_kwargs["eval_set"] = [(X_val, y_val)]
    fit_kwargs["eval_metric"] = "l2"

    # 兼容新旧 LightGBM
    if hasattr(lgb, "early_stopping"):
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))
    else:
        fit_kwargs["early_stopping_rounds"] = early_stopping_rounds

    if hasattr(lgb, "log_evaluation"):
        callbacks.append(lgb.log_evaluation(period=verbose_eval))

    if callbacks:
        fit_kwargs["callbacks"] = callbacks

    model.fit(X_train, y_train, **fit_kwargs)

    best_score = model.best_score_["valid_0"]["l2"]
    best_iter = getattr(model, "best_iteration_", None)

    # 保存模型
    model_path = os.path.join(save_dir, f"fold_{fold + 1}.pkl")
    joblib.dump(model, model_path)

    # 通过 queue 返回日志（主进程统一打印，避免日志乱序）
    logger_queue.put({"fold": fold + 1, "rmse": np.sqrt(best_score), "best_iter": best_iter, "model_path": model_path})

    return model  # 返回模型对象（通过 pipe 或 shared memory 传回主进程）


def train_lightgbm_model(
    model: lgb.LGBMRegressor,
    train_df: pd.DataFrame,
    logger,
    model_save_dir: str,
    period_index: int,
    project_name: str,
    k_folds: int = 4,
    early_stopping_rounds: int = 50,
    verbose_eval: int = 50,
    random_state: int = 42,
    use_gpu: bool = True,
    timestamp: str = None,
    feature_cols: list = None,
) -> List[lgb.LGBMRegressor]:
    """
    并行训练 k-fold LightGBM 模型，每折独占一张 GPU（默认 k_folds=4）
    """
    assert k_folds <= 4, "You have only 4 GPUs"
    if use_gpu:
        assert lgb.__version__ >= "3.3.0", "GPU training requires LightGBM >= 3.3.0"

    start_ts = time.time()
    save_dir = os.path.join(model_save_dir, f"{project_name}_{timestamp}", f"period_{period_index}_k{k_folds}")
    os.makedirs(save_dir, exist_ok=True)

    X = train_df[feature_cols]
    y = train_df["target"]

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    splits = list(kf.split(X))

    # 基础模型参数（不包含 device 等运行时参数）
    base_params = model.get_params()
    if use_gpu:
        base_params.update({"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0})

    logger.info(f"Starting parallel {k_folds}-fold training on {k_folds} GPUs (GPU {'enabled' if use_gpu else 'disabled'})")

    # 多进程通信
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    logger_queue = manager.Queue()

    # 启动并行训练
    train_func = partial(
        _train_single_fold,
        X_train_full=X,
        y_train_full=y,
        base_model_params=base_params,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval,
        save_dir=save_dir,
        logger_queue=logger_queue,
    )

    processes = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        p = mp.Process(target=train_func, args=(fold, train_idx, val_idx))
        p.start()
        processes.append(p)

    # 主进程实时接收日志（顺序更清晰）
    completed = 0
    fold_results = []
    while completed < k_folds:
        msg = logger_queue.get()
        logger.info(
            f"Fold {msg['fold']}/{k_folds} | " f"Val RMSE: {msg['rmse']:.6f} | " f"Best iter: {msg['best_iter'] or 'N/A'} | " f"Saved: {msg['model_path']}"
        )
        fold_results.append(msg)
        completed += 1

    for p in processes:
        p.join()

    # 加载所有模型返回
    models = []
    for msg in fold_results:
        model = joblib.load(msg["model_path"])
        models.append(model)

    mean_rmse = np.mean([r["rmse"] for r in fold_results])
    std_rmse = np.std([r["rmse"] for r in fold_results])

    logger.info(f"{k_folds}-fold CV RMSE: {mean_rmse:.6f} ± {std_rmse:.6f}")
    logger.info(f"All {k_folds} models saved in {save_dir} (total {time.time()-start_ts:.1f}s)")

    return models
