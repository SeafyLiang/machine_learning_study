#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   GPR_Regression.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/1/10 16:53   SeafyLiang   1.0        GPR回归：高斯过程回归
"""
import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import random

'''
sklearn-docs: https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
'''


def method1():
    '''
    参考资料：https://www.zhihu.com/question/445827870/answer/1745254939
    '''
    a = np.random.random(50).reshape(50, 1)
    b = a * 2 + np.random.random(50).reshape(50, 1)
    plt.scatter(a, b, marker='o', color='r', label='3', s=15)
    plt.show()
    gaussian = GaussianProcessRegressor()
    fiting = gaussian.fit(a, b)
    c = np.linspace(0.1, 1, 100)
    d = gaussian.predict(c.reshape(100, 1))
    plt.scatter(a, b, marker='o', color='r', label='3', s=15)
    plt.plot(c, d)
    plt.show()


def f(x):
    """The function to predict."""
    return x * np.sin(x)


def method2():
    '''
    参考资料：https://blog.csdn.net/wong2016/article/details/86500234?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-2.no_search_link&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-2.no_search_link&utm_relevant_index=4
    '''
    np.random.seed(1)

    # ----------------------------------------------------------------------
    #  First the noiseless case
    X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T

    # Observations
    y = f(X).ravel()

    # Mesh the input space for evaluations of the real function, the prediction and
    # its MSE
    x = np.atleast_2d(np.linspace(0, 10, 1000)).T

    # Instantiate a Gaussian Process model
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X, y)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, sigma = gp.predict(x, return_std=True)

    # Plot the function, the prediction and the 95% confidence interval based on
    # the MSE
    plt.figure()
    plt.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
    plt.plot(X, y, 'r.', markersize=10, label=u'Observations')
    plt.plot(x, y_pred, 'b-', label=u'Prediction')
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([y_pred - 1.9600 * sigma,
                             (y_pred + 1.9600 * sigma)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(-10, 20)
    plt.legend(loc='upper left')

    # ----------------------------------------------------------------------
    # now the noisy case
    X = np.linspace(0.1, 9.9, 20)
    X = np.atleast_2d(X).T

    # Observations and noise
    y = f(X).ravel()
    dy = 0.5 + 1.0 * np.random.random(y.shape)
    noise = np.random.normal(0, dy)
    y += noise

    # Instantiate a Gaussian Process model
    gp = GaussianProcessRegressor(kernel=kernel, alpha=dy ** 2,
                                  n_restarts_optimizer=10)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X, y)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, sigma = gp.predict(x, return_std=True)

    # Plot the function, the prediction and the 95% confidence interval based on
    # the MSE
    plt.figure()
    plt.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
    plt.errorbar(X.ravel(), y, dy, fmt='r.', markersize=10, label=u'Observations')
    plt.plot(x, y_pred, 'b-', label=u'Prediction')
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([y_pred - 1.9600 * sigma,
                             (y_pred + 1.9600 * sigma)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(-10, 20)
    plt.legend(loc='upper left')

    plt.show()


def method3():
    '''
    参考资料：https://blog.csdn.net/aaakirito/article/details/117123227?spm=1001.2101.3001.6650.4&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-4.no_search_link&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-4.no_search_link&utm_relevant_index=6
    '''
    '''
        高斯过程回归，Kernel为径向基核函数
    '''
    # 生成高斯过程回归模型
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (0.5, 2))  # 常数核*径向基核函数
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    # 生成历史数据
    xobs = np.array([1, 1.5, -3]).reshape(-1, 1)
    yobs = np.array([3, 0, 1])

    # 使用历史数据拟合模型
    gp.fit(xobs, yobs)

    # 预测
    x_set = np.arange(-6, 6, 0.1).reshape(-1, 1)
    means, sigmas = gp.predict(x_set, return_std=True)

    # 作图
    plt.errorbar(x_set, means, yerr=sigmas, alpha=0.5)
    plt.plot(x_set, means, 'g', linewidth=4)
    colors = ['g', 'r', 'b', 'k']
    for c in colors:
        y_set = gp.sample_y(x_set, random_state=np.random.randint(1000))
        plt.plot(x_set, y_set, c + '--', alpha=0.5)

    plt.show()


if __name__ == '__main__':
    # method1()
    # method2()
    method3()
