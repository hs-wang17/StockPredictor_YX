from datetime import datetime
from dateutil.relativedelta import relativedelta

def generate_train_predict_dates(date_list, train_period_years=3, predict_period_months=3, slide_period_months=3):
    """
    生成训练周期和预测周期的日期列表，从 date_list 中选取真实存在的交易日。
    
    Parameters:
        date_list (list): 交易日列表，按日期升序排列
        train_period_years (int): 训练周期的年数
        predict_period_months (int): 预测周期的月数
        slide_period_months (int): 滑动窗口的月数
        
    Returns:
        tuple: (len(train_dates_list), train_dates_list, predict_dates_list)
            - len(train_dates_list): 训练周期或预测周期的数量
            - train_dates_list: 所有训练周期的日期列表，每个训练周期的日期是一个子列表
            - predict_dates_list: 所有预测周期的日期列表，每个预测周期的日期是一个子列表
    """
    train_dates_list = []
    predict_dates_list = []

    date_format = "%Y%m%d"
    # 将字符串转 datetime
    date_list_dt = [datetime.strptime(d, date_format) for d in date_list]
    
    start_idx = 0
    while start_idx < len(date_list_dt):
        # 计算当前训练周期的开始和结束日期
        train_start = datetime(date_list_dt[start_idx].year, date_list_dt[start_idx].month, 1)
        train_end = train_start + relativedelta(years=train_period_years)
        predict_start = train_end
        predict_end = predict_start + relativedelta(months=predict_period_months)

        # 找出训练周期中在 date_list 中存在的日期
        train_dates = [d.strftime(date_format) for d in date_list_dt if train_start <= d < train_end]
        predict_dates = [d.strftime(date_format) for d in date_list_dt if predict_start <= d < predict_end]

        # 如果预测周期的最后日期超出了日期列表的最后一天
        if predict_dates and datetime.strptime(predict_dates[-1], date_format) > date_list_dt[-1]:
            # 截取预测周期日期，确保不超过最后一个有效日期
            predict_dates = [d for d in predict_dates if datetime.strptime(d, date_format) <= date_list_dt[-1]]

        # 如果预测周期没有足够的日期，表示当前周期有效，后续周期不再生成
        if not predict_dates:
            break

        # 记录训练日期
        train_dates_list.append(train_dates)
        # 记录预测日期
        predict_dates_list.append(predict_dates)

        # 滑动窗口，继续下一周期
        start_idx = next((i for i, d in enumerate(date_list_dt) if d >= train_start + relativedelta(months=slide_period_months)), len(date_list_dt))

    # 返回训练周期和预测周期的数量及对应的日期列表
    return len(train_dates_list), train_dates_list, predict_dates_list
