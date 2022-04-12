#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   series_transformation.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/4/12 23:21   SeafyLiang   1.0       不平稳时间序列转换为平稳的时间序列
"""
import pandas as pd
import matplotlib.pyplot as plt
# 单位根检验法
import statsmodels.tsa.stattools as ts

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
'''
一个平稳的时间序列指的是，未来的时间序列，它的均值、方差和协方差必定与现在的时间序列相等。
若时间序列不平稳，则需要通过差分等技术手段，把它转换成平稳的时间序列，然后再进行预测。
'''
data = pd.read_csv('data/时间序列预测.csv', encoding='utf8', engine='python')

# 设置索引为时间格式
data.index = pd.to_datetime(data.date, format='%Y%m%d')
# 删除date列，已经保存到索引中了
del data['date']
plt.figure()
plt.plot(data, 'r')
plt.show()


# 使用单位根检验法检验是否是平稳的时间序列
# 封装一个方法，方便解读adfuller函数的结果
def tagADF(t):
    result = pd.DataFrame(index=[
        "Test Statistic Value",
        "p-value", "Lags Used",
        "Number of Observations Used",
        "Critical Value(1%)",
        "Critical Value(5%)",
        "Critical Value(10%)"
    ], columns=['value']
    )
    result['value']['Test Statistic Value'] = t[0]
    result['value']['p-value'] = t[1]
    result['value']['Lags Used'] = t[2]
    result['value']['Number of Observations Used'] = t[3]
    result['value']['Critical Value(1%)'] = t[4]['1%']
    result['value']['Critical Value(5%)'] = t[4]['5%']
    result['value']['Critical Value(10%)'] = t[4]['10%']
    return result


# 使用ADF单位根检验法，检验时间序列的稳定性
adf_data = ts.adfuller(data.value)

# 解读ADF单位根检验结果
adfResult = tagADF(adf_data)
print(adfResult)
#                                 value
# Test Statistic Value         -1.16364
# p-value                      0.689038
# Lags Used                          12
# Number of Observations Used        77
# Critical Value(1%)          -3.518281
# Critical Value(5%)          -2.899878
# Critical Value(10%)         -2.587223

# 68%的概率不是一个平稳的时间序列，故通过差分转换成平稳的时间序列
# 对数据进行差分，删除第一个位置的空值
diff = data.value.diff(1).dropna()

plt.figure()
plt.plot(diff, 'r')
plt.show()

# 使用ADF单位根检验法，检验时间序列的稳定性
adf_diff = ts.adfuller(diff)

# 解读ADF单位根检验结果
adfResult = tagADF(adf_diff)
print(adfResult)
#                                 value
# Test Statistic Value        -4.993859
# p-value                      0.000023
# Lags Used                          12
# Number of Observations Used        76
# Critical Value(1%)          -3.519481
# Critical Value(5%)          -2.900395
# Critical Value(10%)         -2.587498

# 0.000023<0.05，认为差分后的时间序列是平稳的时间序列
