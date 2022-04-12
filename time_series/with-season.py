#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   with-season.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/4/12 23:03   SeafyLiang   1.0       季节性时间序列-分解为三部分
"""
import pandas as pd
import statsmodels.api
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']


data = pd.read_csv('data/季节性时间序列.csv', encoding='utf8', engine='python')

total_sales = data['总销量'].values

# 执行季节性时间序列分解
'''
statsmodels.api.tsa.seasonal_decompose(x, freq=None)
x：要分解的季节性时间序列
freq：时间序列的周期，需要自己根据业务来判断数据的周期
'''
tsr = statsmodels.api.tsa.seasonal_decompose(
    total_sales, period=7
)
# 获取趋势部分
trend = tsr.trend
# 获取季节性部分
seasonal = tsr.seasonal
# 获取随机误差部分
random_error = tsr.resid

# 生成2*2的子图
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

ax1.set_title('总销量')
ax1.plot(
    data.index, total_sales, 'k-'
)

ax2.set_title('趋势部分')
ax2.plot(
    data.index, trend, 'g-'
)

ax3.set_title('季节性部分')
ax3.plot(
    data.index, seasonal, 'r-'
)

ax4.set_title('随机误差')
ax4.plot(
    data.index, random_error, 'b-'
)
plt.show()