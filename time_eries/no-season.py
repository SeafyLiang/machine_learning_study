#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   no-season.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/4/12 22:50   SeafyLiang   1.0        非季节性时间序列-SMA和WMA
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data/非季节性时间序列.csv', encoding='utf8', engine='python')

original_data = data['公司A']


def sma_demo():
    # 将非季节性的时间序列，分解为趋势部分和不规则部分
    # SMA 简单移动平均
    sma_data = original_data.rolling(3).mean()
    plt.plot(
        data.index, original_data, 'k-',
        data.index, sma_data, 'g-'
    )
    plt.show()
    # 平滑后的曲线，波动性基本被消除

    # 加入随机误差
    random_error = original_data - sma_data
    plt.plot(
        data.index, original_data, 'k-',
        data.index, sma_data, 'g-',
        data.index, random_error, 'r-'
    )
    plt.show()
    # 随机误差在0附件波动，均值趋向于0


def wma_demo():
    # WMA 加权移动平均
    # 定义窗口大小
    wl = 3
    # 定义每个窗口值的权重
    ww = [1 / 6, 2 / 6, 3 / 6]

    def wma(window):
        return np.sum(window * ww)

    sma_data = original_data.rolling(wl).aggregate(wma)
    random_error = original_data - sma_data
    plt.plot(
        data.index, original_data, 'k-',
        data.index, sma_data, 'g-',
        data.index, random_error, 'r-'
    )
    plt.show()


if __name__ == '__main__':
    sma_demo()
    wma_demo()
