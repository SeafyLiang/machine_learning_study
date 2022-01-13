#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   seaborn_lmplot.py
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2021-06-03 09:34:30   SeafyLiang   1.0          seaborn实现多种回归-回归模型图lmplot
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
sns.lmplot(x="open",
           y="close",
           hue="Up_Down",
           data=dataset)
plt.show()

'''
局部加权线性回归
'''
sns.lmplot(x="open",
           y="close",
           hue="Up_Down",
           lowess=True,
           data=dataset)
plt.show()
'''
对数线性回归模型
'''
sns.lmplot(x="open",
           y="close",
           hue="Up_Down",
           data=dataset,
           logx=True)
plt.show()
'''
稳健线性回归模型
'''
sns.lmplot(x="open",
           y="volume",
           data=dataset,
           hue="Increase_Decrease",
           col="Increase_Decrease",
           # col|hue控制子图不同的变量species
           col_wrap=2,
           height=4,
           robust=True)
plt.show()
'''
多项式回归
'''
sns.lmplot(x="close",
           y="volume",
           data=dataset,
           hue="Increase_Decrease",
           col="Up_Down",  # col|hue控制子图不同的变量species
           col_wrap=2,  # col_wrap控制每行子图数量
           height=4,  # height控制子图高度
           order=3  # 多项式最高次幂
           )
plt.show()
'''
逻辑回归
'''
# 制作具有性别色彩的自定义调色板
pal = dict(Up="#6495ED", Down="#F08080")
# 买卖随开盘价与涨跌变化
g = sns.lmplot(x="open", y="Buy_Sell", col="Up_Down", hue="Up_Down",
               data=dataset,
               palette=pal,
               y_jitter=.02,  # 回归噪声
               logistic=True)  # 逻辑回归模型
plt.show()
