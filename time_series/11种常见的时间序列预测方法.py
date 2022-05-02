#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   11种常见的时间序列预测方法.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/5/2 21:45   SeafyLiang   1.0          11种常见的时间序列预测方法
"""
'''
参考资料：https://mp.weixin.qq.com/s?__biz=Mzg2ODcxMTEwNw==&mid=2247483968&idx=1&sn=708a1edda5944a3ce1607d7ab1aee0fc&chksm=cea96decf9dee4fab12bb24b5148824499267cb9a559ec1e68a6dc5c82825c95a9f72a9b814b&token=1523296504&lang=zh_CN#rd
'''
from random import random
import warnings
warnings.filterwarnings('ignore')

# 指数平滑Exponential Smoothing
def method1():
    # SES
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    # contrived dataset
    data = [x + random() for x in range(1, 100)]
    # fit model
    model = SimpleExpSmoothing(data)
    model_fit = model.fit()
    # make prediction
    yhat = model_fit.predict(len(data), len(data))
    print(yhat)


# Holt-Winters 法
def method2():
    # HWES
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    # contrived dataset
    data = [x + random() for x in range(1, 100)]
    # fit model
    model = ExponentialSmoothing(data)
    model_fit = model.fit()
    # make prediction
    yhat = model_fit.predict(len(data), len(data))
    print(yhat)


# 自回归 (AR)
def method3():
    # AR
    from statsmodels.tsa.ar_model import AutoReg
    # contrived dataset
    data = [x + random() for x in range(1, 100)]
    # fit model
    model = AutoReg(data, lags=1)
    model_fit = model.fit()
    # make prediction
    yhat = model_fit.predict(len(data), len(data))
    print(yhat)


def method4():
    # MA
    from statsmodels.tsa.arima.model import ARIMA
    # contrived dataset
    data = [x + random() for x in range(1, 100)]
    # fit model
    model = ARIMA(data, order=(0, 0, 1))
    model_fit = model.fit()
    # make prediction
    yhat = model_fit.predict(len(data), len(data))
    print(yhat)


# 自回归滑动平均模型 (ARMA)
def method5():
    # ARMA
    from statsmodels.tsa.arima.model import ARIMA
    # contrived dataset
    data = [random() for x in range(1, 100)]
    # fit model
    model = ARIMA(data, order=(2, 0, 1))
    model_fit = model.fit()
    # make prediction
    yhat = model_fit.predict(len(data), len(data))
    print(yhat)


# 差分整合移动平均自回归模型 (ARIMA)
def method6():
    # ARIMA
    from statsmodels.tsa.arima.model import ARIMA
    # contrived dataset
    data = [x + random() for x in range(1, 100)]
    # fit model
    model = ARIMA(data, order=(1, 1, 1))
    model_fit = model.fit()
    # make prediction
    yhat = model_fit.predict(len(data), len(data), typ='levels')
    print(yhat)


# 季节性 ARIMA  (SARIMA)
def method7():
    # SARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    # contrived dataset
    data = [x + random() for x in range(1, 100)]
    # fit model
    model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
    model_fit = model.fit(disp=False)
    # make prediction
    yhat = model_fit.predict(len(data), len(data))
    print(yhat)


# 包含外生变量的SARIMA (SARIMAX)
def method8():
    # SARIMAX
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    # contrived dataset
    data1 = [x + random() for x in range(1, 100)]
    data2 = [x + random() for x in range(101, 200)]
    # fit model
    model = SARIMAX(data1, exog=data2, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
    model_fit = model.fit(disp=False)
    # make prediction
    exog2 = [200 + random()]
    yhat = model_fit.predict(len(data1), len(data1), exog=[exog2])
    print(yhat)


# 向量自回归 (VAR)
def method9():
    # VAR
    from statsmodels.tsa.vector_ar.var_model import VAR
    # contrived dataset with dependency
    data = list()
    for i in range(100):
        v1 = i + random()
        v2 = v1 + random()
        row = [v1, v2]
        data.append(row)
    # fit model
    model = VAR(data)
    model_fit = model.fit()
    # make prediction
    yhat = model_fit.forecast(model_fit.y, steps=1)
    print(yhat)


# 向量自回归滑动平均模型  (VARMA)
def method10():
    # VARMA
    from statsmodels.tsa.statespace.varmax import VARMAX
    # contrived dataset with dependency
    data = list()
    for i in range(100):
        v1 = random()
        v2 = v1 + random()
        row = [v1, v2]
        data.append(row)
    # fit model
    model = VARMAX(data, order=(1, 1))
    model_fit = model.fit(disp=False)
    # make prediction
    yhat = model_fit.forecast()
    print(yhat)


# 包含外生变量的向量自回归滑动平均模型 (VARMAX)
def method11():
    # VARMAX
    from statsmodels.tsa.statespace.varmax import VARMAX
    # contrived dataset with dependency
    data = list()
    for i in range(100):
        v1 = random()
        v2 = v1 + random()
        row = [v1, v2]
        data.append(row)
    data_exog = [x + random() for x in range(100)]
    # fit model
    model = VARMAX(data, exog=data_exog, order=(1, 1))
    model_fit = model.fit(disp=False)
    # make prediction
    data_exog2 = [[100]]
    yhat = model_fit.forecast(exog=data_exog2)
    print(yhat)


if __name__ == '__main__':
    method1()  # 指数平滑Exponential Smoothing
    method2()  # Holt-Winters 法
    method3()  # 自回归 (AR)
    method4()  # 移动平均模型(MA)
    method5()  # 自回归滑动平均模型 (ARMA)
    method6()  # 差分整合移动平均自回归模型 (ARIMA)
    method7()  # 季节性 ARIMA  (SARIMA)
    method8()  # 包含外生变量的SARIMA (SARIMAX)
    method9()  # 向量自回归 (VAR)
    method10()  # 向量自回归滑动平均模型  (VARMA)
    method11()  # 包含外生变量的向量自回归滑动平均模型 (VARMAX)
