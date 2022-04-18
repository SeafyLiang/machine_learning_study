#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   ols_regression.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/4/18 18:27   SeafyLiang   1.0       ols回归及置信区间
"""
# 基于波士顿数据集实现线性回归demo
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# 加载数据集
from statsmodels.sandbox.regression.predstd import wls_prediction_std

data = load_boston()
data_df = pd.DataFrame(data.data, columns=data.feature_names)
target_df = pd.DataFrame(data.target, columns=['Target'])

data_df = data_df[['RM']]
print(data_df.head())

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(data_df, target_df, test_size=0.2, random_state=125)

# 2、OLS回归分析报告
import statsmodels.api as sm
# 将特征变量x_test添加常数项，赋值给x2
x2 = sm.add_constant(x_test)
# 对y_test和x2，用OLS()最小二乘法，进行线性回归方程搭建
est = sm.OLS(y_test, x2).fit()
# 打印该模型的数据信息
print(est.summary())
# R-squared越接近1，模型拟合程度越高。
# Adj. R-squared是R-squared的改进版。
# p值小于0.05，认为与目标变量有显著相关性。


# 获取置信区间
# wls_prediction_std(housing_model)返回三个值, 标准差，置信区间下限，置信区间上限
_, confidence_interval_lower, confidence_interval_upper = wls_prediction_std(est)
# 创建画布
fig, ax = plt.subplots(figsize=(10,7))
# 'bo' 代表蓝色的圆形,
ax.plot(x_test, y_test, 'bo', label="real")
# 绘制趋势线
ax.plot(x_test, est.fittedvalues, 'g--.', label="OLS")
# 绘制上下置信区间
ax.plot(x_test, confidence_interval_upper, 'r--')
ax.plot(x_test, confidence_interval_lower, 'r--')
# 绘制标题，网格，和图例
ax.set_title('ols_regression')
ax.grid()
ax.legend(loc='best')
plt.show()

df_result = pd.DataFrame()
df_result['x'] = x_test
df_result['confidence_interval_lower'] = confidence_interval_lower
df_result['y_pred'] = est.fittedvalues
df_result['confidence_interval_upper'] = confidence_interval_upper

print(df_result.head())