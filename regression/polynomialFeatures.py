#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   polynomialFeatures.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/4/1 17:49   SeafyLiang   1.0         多项式回归实现非线性回归
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures as PF, StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


def demo1():
    '''
    利用多项式将数据升维，并拟合数据
    '''
    rnd = np.random.RandomState(42)  # 设置随机数种子
    X = rnd.uniform(-3, 3, size=100)
    y = np.sin(X) + rnd.normal(size=len(X)) / 3
    # 将X升维，准备好放入sklearn中
    X = X.reshape(-1, 1)
    # 多项式拟合，设定高次项
    d = 5
    # 原始特征矩阵的拟合结果
    LinearR = LinearRegression().fit(X, y)
    # 进行高此项转换
    X_ = PF(degree=d).fit_transform(X)
    LinearR_ = LinearRegression().fit(X_, y)
    line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
    line_ = PF(degree=d).fit_transform(line)
    # 放置画布
    fig, ax1 = plt.subplots(1)
    # 将测试数据带入predict接口，获得模型的拟合效果并进行绘制
    ax1.plot(line, LinearR.predict(line), linewidth=2, color='green'
             , label="linear regression")
    ax1.plot(line, LinearR_.predict(line_), linewidth=2, color='orange'
             , label="Polynomial regression")  # 将原数据上的拟合绘制在图像上
    ax1.plot(X[:, 0], y, 'o', c='k')
    # 其他图形选项
    ax1.legend(loc="best")
    ax1.set_ylabel("Regression output")
    ax1.set_xlabel("Input feature")
    ax1.set_title("Linear Regression ordinary vs poly")
    ax1.text(0.8, -1, f"LinearRegression score:\n{LinearR.score(X, y)}", fontsize=15)
    ax1.text(0.8, -1.3, f"LinearRegression score after poly :\n{LinearR_.score(X_, y)}", fontsize=15)

    plt.tight_layout()
    plt.show()


# 生产数据函数
def uniform(size):
    x = np.linspace(0, 1, size)
    return x.reshape(size, 1)


def create_data(size):
    x = uniform(size)
    np.random.seed(42)  # 设置随机数种子
    y = sin_fun(x) + np.random.normal(scale=0.25, size=x.shape)
    return x, y


def sin_fun(x):
    return np.sin(2 * np.pi * x)


def demo2():
    '''
    不同的最高次取值，对模型拟合效果有重要的影响。
    '''
    X_train, y_train = create_data(20)
    X_test = uniform(200)
    y_test = sin_fun(X_test)
    fig = plt.figure(figsize=(12, 8))
    for i, degree in enumerate([0, 1, 3, 6, 9, 12]):
        plt.subplot(2, 3, i + 1)

        poly = PF(degree)
        X_train_ploy = poly.fit_transform(X_train)
        X_test_ploy = poly.fit_transform(X_test)

        lr = LinearRegression()
        lr.fit(X_train_ploy, y_train)
        y_pred = lr.predict(X_test_ploy)

        plt.scatter(X_train, y_train, facecolor="none", edgecolor="g", s=25, label="training data")
        plt.plot(X_test, y_pred, c="orange", label="fitting")
        plt.plot(X_test, y_test, c="k", label="$\sin(2\pi x)$")
        plt.title("N={}".format(degree))
        plt.legend(loc="best")
        plt.ylabel("Regression output")
        #     plt.xlabel("Input feature")
        plt.legend()
    plt.show()


def demo3():
    '''
    利用pipeline将三个模型封装起来串联操作，让模型接口更加简洁，使用起来方便
    '''
    X, y = create_data(200)  # 利用上面的生产数据函数
    degree = 6
    # 利用Pipeline将三个模型封装起来串联操作n
    poly_reg = Pipeline([
        ("poly", PF(degree=degree)),
        ("std_scaler", StandardScaler()),
        ("lin_reg", LinearRegression())
    ])
    fig = plt.figure(figsize=(10, 6))
    poly = PF(degree)
    poly_reg.fit(X, y)
    y_pred = poly_reg.predict(X)
    # 可视化结果
    plt.scatter(X, y, facecolor="none", edgecolor="g", s=25, label="training data")
    plt.plot(X, y_pred, c="orange", label="fitting")
    # plt.plot(X,y,c="k",label="$\sin(2\pi x)$")
    plt.title("degree={}".format(degree))
    plt.legend(loc="best")
    plt.ylabel("Regression output")
    plt.xlabel("Input feature")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    demo1()
    demo2()
    demo3()
