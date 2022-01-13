#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   Linear_Regression.py
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/1/13 19:56   SeafyLiang   1.0       多元线性回归、一元二次线性回归，p值与R2
"""
# 基于波士顿数据集实现线性回归demo
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# 加载数据集
data = load_boston()
data_df = pd.DataFrame(data.data, columns=data.feature_names)
target_df = pd.DataFrame(data.target, columns=['Target'])

print(data_df.head())

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(data_df, target_df, test_size=0.2, random_state=125)

# 1、多元一次线性回归
lr_clf = LinearRegression().fit(x_train, y_train)
y_pred = lr_clf.predict(x_test)

# 绘图比较结果
plt.plot(range(y_test.shape[0]), y_test, color='blue', linewidth=1.5, linestyle='-')
plt.plot(range(y_pred.shape[0]), y_pred, color='red', linewidth=1.5, linestyle='-')
plt.legend(['act', 'pred'])
plt.show()

# mse：均方误差
print("Mean squared error:", mean_squared_error(y_pred=y_pred, y_true=y_test))

# 模型训练的系数
corr_df = pd.DataFrame(data.feature_names, columns=['Features'])
corr_df['weight'] = lr_clf.coef_[0]
print(corr_df)

# 2、一元二次线性回归
# 引入多次项内容模块
from sklearn.preprocessing import PolynomialFeatures

# 设置最高次项为2次
poly_reg = PolynomialFeatures(degree=2)

data_df = data_df[['NOX']]

print(data_df.head())

# 将原有数据转换为新的二维数组
x_train_2 = poly_reg.fit_transform(x_train)
x_test_2 = poly_reg.fit_transform(x_test)
lr_2_clf = LinearRegression()
lr_2_clf.fit(x_train_2, y_train)

y_pred_2 = lr_2_clf.predict(x_test_2)

# 绘图比较结果
plt.plot(range(y_test.shape[0]), y_test, color='blue', linewidth=1.5, linestyle='-')
plt.plot(range(y_pred_2.shape[0]), y_pred_2, color='red', linewidth=1.5, linestyle='-')
plt.legend(['act', 'pred'])
plt.show()

# mse：均方误差
print("Mean squared error:", mean_squared_error(y_pred=y_pred_2, y_true=y_test))

# 模型训练的系数
print(lr_2_clf.coef_)  # 获取系数a, b
print(lr_2_clf.intercept_)  # 获取常数项c

# 线性回归模型评估
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

# 假设检验p值
from scipy import stats

rvs1 = stats.norm.rvs(loc=5, scale=10, size=500)
rvs2 = stats.norm.rvs(loc=5, scale=10, size=500)
print(stats.ttest_ind(rvs1, rvs2))

# R-squared另一种获取方法
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print('r2_score:', r2)  # 拟合程度，（非过拟合状态下）越接近1越好
