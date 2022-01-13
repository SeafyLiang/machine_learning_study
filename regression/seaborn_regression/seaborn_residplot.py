#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   seaborn_residplot.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2021-06-03 13:34:52   SeafyLiang   1.0          seaborn实现多种回归-线性回归图residplot
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
线性回归的残差
'''
# 拟合线性模型后绘制残差,lowess平滑
x=dataset.open
y=dataset.Returns
sns.residplot(x=x, y=y,
              lowess=True,
              color="g")
plt.show()
'''
稳健回归残差图
'''
sns.residplot(x="open",
              y="Returns",
              data=dataset.loc[dataset.Up_Down == "Up"],
              robust=True,
              lowess=True)
plt.show()
'''
多项式回归残差图
'''
sns.residplot(x="open",
              y="close",
              data=dataset.loc[dataset.Up_Down == "Up"],
              order=3,
              lowess=True)
plt.show()
