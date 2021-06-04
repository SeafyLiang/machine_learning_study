#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   seaborn_other.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2021-06-03 10:34:35  SeafyLiang   1.0          seaborn实现多种回归-其他背景中添加回归
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
jointplot
'''
sns.jointplot("open",
              "Returns",
              data=dataset,
              kind='reg')
# 设置kind="reg"为添加线性回归拟合
# （使用regplot()）和单变量KDE曲线
plt.show()
# jointplot()可以通过kind="resid"来调用residplot()绘制具有单变量边际分布。
sns.jointplot(x="open",
              y="close",
              data=dataset,
              kind="resid")
plt.show()

'''
pairplot
给pairplot()传入kind="reg"参数则会融合regplot()与PairGrid来展示变量间的线性关系。注意这里和lmplot()的区别，lmplot()绘制的行（或列）是将一个变量的多个水平（分类、取值）展开，而在这里，PairGrid则是绘制了不同变量之间的线性关系。
'''
sns.pairplot(dataset,
             x_vars=["open", "close"],
             y_vars=["Returns"],
             height=5,
             aspect=.8,
             kind="reg");
plt.show()
