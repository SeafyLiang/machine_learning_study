#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   timeSeriesPrediction.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2021/7/16 15:28   SeafyLiang   1.0          时序预测
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Subsetting the dataset
# Index 11856 marks the end of year 2013
df = pd.read_csv('https://gitee.com/myles2019/dataset/raw/master/jetrail/jetrail_train.csv', nrows=11856)

# Creating train and test set
# Index 10392 marks the end of October 2013
train = df[0:10500]
test = df[10500:10700]

# Aggregating the dataset at daily level
df.Timestamp = pd.to_datetime(df.Datetime, format='%d-%m-%Y %H:%M')
df.index = df.Timestamp
df = df.resample('D').mean()

train.Timestamp = pd.to_datetime(train.Datetime, format='%d-%m-%Y %H:%M')
train.index = train.Timestamp
train = train.resample('D').mean()

test.Timestamp = pd.to_datetime(test.Datetime, format='%d-%m-%Y %H:%M')
test.index = test.Timestamp
test = test.resample('D').mean()

# Plotting data
train.Count.plot(figsize=(15, 8), title='Daily Ridership', fontsize=14)
test.Count.plot(figsize=(15, 8), title='Daily Ridership', fontsize=14)
plt.show()

# 简单方法预测
"""
有时候整个时段的时间序列值是稳定的，如果想预测未来1天的值，我们只需要用1天前的数据作为预估值。这种预估方法的核心假设是未来的数据和最新观测的数据是一样的
"""
dd = np.asarray(train.Count)
y_hat = test.copy()
y_hat['naive'] = dd[len(dd) - 1]
plt.figure(figsize=(12, 8))
plt.plot(train.index, train['Count'], label='Train')
plt.plot(test.index, test['Count'], label='Test')
plt.plot(y_hat.index, y_hat['naive'], label='Naive Forecast')
plt.legend(loc='best')
plt.title("Naive Forecast")
plt.show()

from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(test.Count, y_hat.naive))
print(rms)

# RMSE = 43.9164061439
"""
简单预测方法并不适合波动很大的数据集，仅适合比较稳定的数据集。
"""

# ARIMA（效果较好）
"""
另一个常用的时间序列预测模型是非常流行的，就是ARIMA模型，是Autoregressive Integrated Moving average的缩写。

指数平滑法是基于数据的趋势项、季节项的描述来做预测，而ARIMA旨在发现数据之间相互的相关性来预测。一种ARIMA的改进变体是SARIMA（Seasonal ARIMA），考虑了季节性因素。
"""
import statsmodels.api as sm

y_hat_avg = test.copy()
fit1 = sm.tsa.statespace.SARIMAX(train.Count, order=(2, 1, 4), seasonal_order=(0, 1, 1, 7)).fit()
y_hat_avg['SARIMA'] = fit1.predict(start="2013-10-1", end="2013-11-30", dynamic=True)
plt.figure(figsize=(16, 8))
plt.plot(train['Count'], label='Train')
plt.plot(test['Count'], label='Test')
plt.plot(y_hat_avg['SARIMA'], label='SARIMA')
plt.legend(loc='best')
plt.show()
