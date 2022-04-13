#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   time_series_predict.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/4/13 22:18   SeafyLiang   1.0       时序预测-ar,sma,arma
"""
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

data = pd.read_csv('data/时间序列预测.csv', encoding='utf8', engine='python')

# 设置索引为时间格式
data.index = pd.to_datetime(data.date, format='%Y%m%d')
# 删除date列，已经保存到索引中了
del data['date']

diff = data.value.diff(1).dropna()

plt.figure()
plt.plot(diff, 'r')
plt.show()


def ar_demo():
    '''
    自回归
    :return:
    '''
    # 使用 AR 建模
    arModel = sm.tsa.AR(
        diff
    )

    # 使用 AR 模型的 select_order 方法
    # 自动根据 AIC 指标，选择最优的 p 值
    p = arModel.select_order(
        maxlag=30,
        ic='aic'
    )
    print(p)

    # 使用最优的 p 值进行建模
    arModel = arModel.fit(maxlag=p)

    # 对时间序列模型进行评估
    delta = arModel.fittedvalues - diff
    score = 1 - delta.var() / diff.var()
    print(score)


def sma_demo():
    '''
    简单移动平均
    :return:
    '''
    sma_data = diff.rolling(3).mean()
    plt.plot(
        diff.index, diff, 'k-',
        diff.index, sma_data, 'g-'
    )
    plt.show()


def arma_demo():
    # ARMA模型，使用AIC指标
    # AR模型从1-15中选择最优的p
    # MA模型从1-15中选择最优的q
    # 执行时间非常长，作者执行了10个小时左右
    # ic = sm.tsa.arma_order_select_ic(
    #     diff,
    #     max_ar=15,
    #     max_ma=15,
    #     ic='aic'
    # )
    # 选择最优参数
    order = (6, 6)

    armaModel = sm.tsa.ARMA(
        diff, order
    ).fit()
    # 评估模型得分
    delta = armaModel.fittedvalues - diff
    score = 1 - delta.var() / diff.var()
    print(score)

    # 预测接下来10天的值
    p = armaModel.predict(
        start='2016-03-31',
        end='2016-04-10'
    )

    # 封装一个对差分的数据进行还原的函数
    def revert(diffValues, *lastValue):
        for i in range(len(lastValue)):
            result = []
            lv = lastValue[i]
            for dv in diffValues:
                lv = dv + lv
                result.append(lv)
            diffValues = result
        return diffValues

    # 对差分的数据进行还原
    r = revert(p, 10395)

    plt.figure()
    plt.plot(
        diff, 'r',
        label='Raw'
    )
    plt.plot(
        armaModel.fittedvalues, 'g',
        label='ARMA Model'
    )
    plt.plot(
        p, 'b', label='ARMA Predict'
    )
    plt.legend()
    plt.show()

    r = pd.DataFrame({
        'value': r
    }, index=p.index
    )
    plt.figure()
    plt.plot(
        data.value, c='r',
        label='Raw'
    )
    plt.plot(
        r, c='g',
        label='ARMA Model'
    )
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # ar_demo()
    # sma_demo()
    arma_demo()