import config.config as config
import pipeline.data as pipeline_data
import pipeline.filter as pipeline_filter
import pipeline.predict_ensemble as pipeline_predict_ensemble
import pipeline.train_ensemble as pipeline_train_ensemble
import utils.ensemble_model as utils_ensemble_model
import utils.function as utils_function
import utils.logger as utils_logger

import numpy as np
import pandas as pd
import os
import tqdm

def run():
    # Setup logger and args
    args, logger = config.load_config_with_logger()

    # Generate periods (与你原来相同的 helper)
    date_list = sorted([file_name[:8] for file_name in os.listdir(args.data_dir)])
    num_periods, train_dates_list, predict_dates_list = utils_function.generate_train_predict_dates(
        date_list,
        train_period_days=args.train_period_days,
        predict_period_days=args.predict_period_days,
        slide_period_days=args.slide_period_days,
        gap_days=args.gap_days
    )

    logger.info(f"Number of periods: {num_periods}")
    logger.info("=" * 60)

    all_predictions_list = []

    for i in range(num_periods)[:2]:   # 这里保留你原来的 [:2] 限制，必要时移除
        logger.info(f"Period {i+1}:")
        logger.info(f"  Train Dates: {train_dates_list[i][0]} to {train_dates_list[i][-1]}; Length: {len(train_dates_list[i])} days")
        logger.info(f"  Predict Dates: {predict_dates_list[i][0]} to {predict_dates_list[i][-1]}; Length: {len(predict_dates_list[i])} days")

        train_date_list, predict_date_list = train_dates_list[i], predict_dates_list[i]
        train_frames = []
        predict_frames = []
        filter_index = pipeline_filter.read_filter_index(file_path=args.filter_file_path, period_index=i)

        # ------- 加载训练数据（示例只取前100个日期以防爆内存，可按需移除 [:100]） -------
        for date in tqdm.tqdm(train_date_list[:100], desc="Loading training data"):
            file_path = os.path.join(args.data_dir, f"{date}.fea")
            data = pipeline_data.load_data(file_path)
            # 保持你原来处理列名/选择逻辑
            data.columns = ['code'] + [data.columns[j].strip() for j in range(len(data.columns) - 1)]
            stock_code_col, feature_cols = data.columns[0], data.columns[3:15][filter_index]
            data = pd.concat([data[stock_code_col], data[feature_cols]], axis=1)
            data = pipeline_data.ensure_data_types(data)
            data = pipeline_data.fill_missing_values(data, fill_value=0.0)
            data = pipeline_data.normalize_columns(data, feature_cols)
            data.insert(0, 'date', date)

            # target: 你原来用 pipeline_data.load_data(file_path).iloc[:,4] 获取
            target = pipeline_data.load_data(file_path).iloc[:, 4]
            # 合并 code/date/feature/target
            temp = data.copy()
            temp['target'] = target.values
            train_frames.append(temp)

        # ------- 加载预测数据 -------
        for date in tqdm.tqdm(predict_date_list[:100], desc="Loading prediction data"):
            file_path = os.path.join(args.data_dir, f"{date}.fea")
            data = pipeline_data.load_data(file_path)
            data.columns = ['code'] + [data.columns[j].strip() for j in range(len(data.columns) - 1)]
            stock_code_col, feature_cols = data.columns[0], data.columns[3:15][filter_index]
            data = pd.concat([data[stock_code_col], data[feature_cols]], axis=1)
            data = pipeline_data.ensure_data_types(data)
            data = pipeline_data.fill_missing_values(data, fill_value=0.0)
            data = pipeline_data.normalize_columns(data, feature_cols)
            data.insert(0, 'date', date)

            # target (可选，预测时通常没有)
            target = pipeline_data.load_data(file_path).iloc[:, 4]
            temp = data.copy()
            temp['target'] = target.values
            predict_frames.append(temp)

        if len(train_frames) == 0:
            logger.warning(f"No training data for period {i}. Skip.")
            continue

        train_df = pd.concat(train_frames, ignore_index=True)
        predict_df = pd.concat(predict_frames, ignore_index=True) if len(predict_frames) > 0 else pd.DataFrame(columns=train_df.columns)

        # X, y
        feature_cols_list = list(feature_cols)  # 上面循环里定义的 feature_cols
        X_train = train_df[feature_cols_list]
        y_train = train_df['target']

        # 创建 LGBM 模型（你可以在 config 中传 params）
        lgb_params = getattr(args, "lgb_params", None)
        model = utils_ensemble_model.create_lgbm_model(params=lgb_params, n_estimators=args.lgb_n_estimators, random_state=args.random_seed)

        # 训练
        model = pipeline_train_ensemble.train_lightgbm_model(
            model, X_train, y_train, logger,
            model_save_dir=args.model_save_dir,
            period_index=i,
            project_name=args.project_name,
            early_stopping_rounds=args.early_stopping_rounds,
            valid_size=args.valid_size,
            verbose_eval=args.verbose_eval
        )

        # 预测
        if not predict_df.empty:
            preds = pipeline_predict_ensemble.make_predictions_lightgbm(
                model, predict_df, feature_cols_list, logger,
                predictions_save_dir=args.predictions_save_dir,
                project_name=args.project_name,
                period_index=i
            )
            all_predictions_list.append(preds)
        else:
            logger.info("No predict_df for this period. Skipping prediction.")

    # Combine all period predictions
    print(all_predictions_list)
    if len(all_predictions_list) > 0:
        logger.info("Concatenating all period predictions...")
        combined_predictions = pd.concat(all_predictions_list)
        combined_output_path = os.path.join(
            args.predictions_save_dir,
            f"{args.project_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_combined_predictions.csv"
        )
        combined_predictions.to_csv(combined_output_path)
        logger.info(f"All periods combined predictions saved to {combined_output_path}")
    else:
        logger.info("No predictions generated for any period.")

    logger.info("All periods processed.")

if __name__ == "__main__":
    run()
