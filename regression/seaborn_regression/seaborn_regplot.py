#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   seaborn_regplot.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2021-06-03 11:34:46   SeafyLiang   1.0          seaborn实现多种回归-线性回归图regplot
"""
'''
使用股市数据--中国平安sh.601318历史k线数据。
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import baostock as bs

bs.login()
result = bs.query_history_k_data('sh.601318',
                                 fields='date,open,high, low,close,volume',
                                 start_date='2020-01-01',
                                 end_date='2021-05-01',
                                 frequency='d')
dataset = result.get_data().set_index('date').applymap(lambda x: float(x))
bs.logout()
dataset['Open_Close'] = (dataset['open'] - dataset['close']) / dataset['open']
dataset['High_Low'] = (dataset['high'] - dataset['low']) / dataset['low']
dataset['Increase_Decrease'] = np.where(dataset['volume'].shift(-1) > dataset['volume'], 1, 0)
dataset['Buy_Sell_on_Open'] = np.where(dataset['open'].shift(-1) > dataset['open'], 1, 0)
dataset['Buy_Sell'] = np.where(dataset['close'].shift(-1) > dataset['close'], 1, 0)
dataset['Returns'] = dataset['close'].pct_change()
dataset = dataset.dropna()
dataset['Up_Down'] = np.where(dataset['Returns'].shift(-1) > dataset['Returns'], 'Up', 'Down')
dataset = dataset.dropna()
print(dataset.head())
'''
线性回归
'''
# 绘制线性回归拟合曲线
f, ax = plt.subplots(figsize=(8,6))
sns.regplot(x="Returns",
            y="volume",
            data=dataset,
            fit_reg=True,
            ci = 95,
            scatter=True,
            ax=ax)
plt.show()
f, ax = plt.subplots(1,2,figsize=(15,6))
sns.regplot(x="Returns",
            y="volume",
            data=dataset,
            x_bins=10,
            x_ci="ci",
            ax=ax[0])
# 带有离散x变量的图，显示了唯一值的方差和置信区间：
sns.regplot(x="Returns",
            y="volume",
            data=dataset,
            x_bins=10,
            x_ci='sd',
            ax=ax[1])
plt.show()
'''
多项式回归
'''
sns.regplot(x="open",
            y="close",
            data=dataset.loc[dataset.Up_Down == "Up"],
            scatter_kws={"s": 80},
            order=5, ci=None)
plt.show()
'''
逻辑回归
'''
sns.regplot(x= "volume",
            y= "Increase_Decrease",
            data=dataset,
            logistic=True,
            n_boot=500,
            y_jitter=.03,)
plt.show()
'''
对数线性回归
'''
sns.regplot(x="open",
            y="volume",
            data=dataset.loc[dataset.Up_Down == "Up"],
            x_estimator=np.mean,
            logx=True)
plt.show()
'''
稳健线性回归
'''
sns.regplot(x="open",
            y="Returns",
            data=dataset.loc[dataset.Up_Down == "Up"],
            scatter_kws={"s": 80},
            robust=True, ci=None)
plt.show()