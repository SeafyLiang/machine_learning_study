#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   prophet_time_forecast.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2021/8/12 10:57   SeafyLiang   1.0      时间序列预测算法Prophet简易入门
"""
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
"""
Prophet 简介
Prophet是Facebook开源的时间序列预测算法，可以有效处理节假日信息，并按周、月、年对时间序列数据的变化趋势进行拟合。根据官网介绍，Prophet对具有强烈周期性特征的历史数据拟合效果很好，不仅可以处理时间序列存在一些异常值的情况，也可以处理部分缺失值的情形。算法提供了基于Python和R的两种实现方式。
从论文上的描述来看，这个 prophet 算法是基于时间序列分解和机器学习的拟合来做的，其中在拟合模型的时候使用了 pyStan 这个开源工具，因此能够在较快的时间内得到需要预测的结果。

参考资料：https://blog.csdn.net/anshuai_aw1/article/details/83412058

输入已知的时间序列的时间戳和相应的值；
输入需要预测的时间序列的长度；
输出未来的时间序列走势。
输出结果可以提供必要的统计指标，包括拟合曲线，上界和下界等。

"""

# 读入数据集
df = pd.read_csv('data/example_wp_log_peyton_manning.csv')
print(df.head())
# 拟合模型
m = Prophet()
m.fit(df)

# 构建待预测日期数据框，periods = 365 代表除历史数据的日期外再往后推 365 天
future = m.make_future_dataframe(periods=365)
future.tail()
# 预测数据集
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# 展示预测结果
m.plot(forecast)
# 预测的成分分析绘图，展示预测中的趋势、周效应和年度效应
m.plot_components(forecast)
plt.show()

