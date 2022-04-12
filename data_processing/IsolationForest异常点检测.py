#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   IsolationForest异常点检测.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/4/12 10:23   SeafyLiang   1.0          孤立森林异常点检测
"""
'''
Isolation Forest算法的逻辑很直观，算法采用二叉树对数据进行分裂，样本选取、特征选取、分裂点选取都采用随机化的方式
（我称之为瞎胡乱分，诶，但是人家效果出奇的好，你说气人不）。如果某个样本是异常值，可能需要很少次数就可以切分出来。
从统计意义上来说，相对聚集的点需要分割的次数较多，比较孤立的点需要的分割次数少，孤立森林就是利用分割的次数来度量一个点是聚集的（正常）还是孤立的（异常）。

1、前提假设
异常样本不能占比太高
异常样本和正常样本差异较大
2、算法思想
异常样本更容易快速落入叶子结点或者说，异常样本在决策树上，距离根节点更近
3、模型参数
n_estimators : int, optional (default=100)

iTree的个数，指定该森林中生成的随机树数量，默认为100个

max_samples : int or float, optional (default=”auto”)

构建子树的样本数，整数为个数，小数为占全集的比例，用来训练随机数的样本数量，即子采样的大小

如果设置的是一个int常数，那么就会从总样本X拉取max_samples个样本来生成一棵树iTree

如果设置的是一个float浮点数，那么就会从总样本X拉取max_samples * X.shape[0]个样本,X.shape[0]表示总样本个数

如果设置的是"auto"，则max_samples=min(256, n_samples)，n_samples即总样本的数量

如果max_samples值比提供的总样本数量还大的话，所有的样本都会用来构造数，意思就是没有采样了，构造的n_estimators棵iTree使用的样本都是一样的，即所有的样本

contamination : float in (0., 0.5), optional (default=0.1)

取值范围为(0., 0.5),表示异常数据占给定的数据集的比例，数据集中污染的数量，其实就是训练数据中异常数据的数量，比如数据集异常数据的比例。定义该参数值的作用是在决策函数中定义阈值。如果设置为'auto'，则决策函数的阈值就和论文中定义的一样

max_features : int or float, optional (default=1.0)

构建每个子树的特征数，整数位个数，小数为占全特征的比例，指定从总样本X中抽取来训练每棵树iTree的属性的数量，默认只使用一个属性

如果设置为int整数，则抽取max_features个属性

如果是float浮点数，则抽取max_features * X.shape[1]个属性

bootstrap : boolean, optional (default=False)
采样是有放回还是无放回，如果为True，则各个树可放回地对训练数据进行采样。如果为False，则执行不放回的采样。

n_jobs :int or None, optional (default=None)
在运行fit()和predict()函数时并行运行的作业数量。除了在joblib.parallel_backend上下文的情况下，None表示为1。设置为-1则表示使用所有可用的处理器

random_state : int, RandomState instance or None, optional (default=None)

每次训练的随机性

如果设置为int常数，则该random_state参数值是用于随机数生成器的种子

如果设置为RandomState实例，则该random_state就是一个随机数生成器

如果设置为None，该随机数生成器就是使用在np.random中的RandomState实例

verbose : int, optional (default=0)

训练中打印日志的详细程度，数值越大越详细

warm_start : bool, optional (default=False)
当设置为True时，重用上一次调用的结果去fit,添加更多的树到上一次的森林1集合中；否则就fit一整个新的森林

'''
# 加载模型所需要的的包
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.ensemble import IsolationForest


def created_data_od():
    '''
    构造的一维数据做异常点检测
    '''
    # 构造一个数据集，只包含一列数据，假如都是月薪数据，有些可能是错的
    df = pd.DataFrame({'salary': [4, 1, 4, 5, 3, 6, 2, 5, 6, 2, 5, 7, 1, 8, 12, 33, 4, 7, 6, 7, 8, 55]})

    # 构建模型 ,n_estimators=100 ,构建100颗树
    model = IsolationForest(n_estimators=100,
                            max_samples='auto',
                            contamination=float(0.1),
                            max_features=1.0)
    # 训练模型
    model.fit(df[['salary']])

    # 预测 decision_function 可以得出 异常评分
    df['scores'] = model.decision_function(df[['salary']])

    #  predict() 函数 可以得到模型是否异常的判断，-1为异常，1为正常
    df['anomaly'] = model.predict(df[['salary']])
    print(df)
    #     salary    scores  anomaly
    # 0        4  0.200080        1
    # 1        1  0.101944        1
    # 2        4  0.200080        1
    # 3        5  0.216688        1
    # 4        3  0.138629        1
    # 5        6  0.228297        1
    # 6        2  0.146017        1
    # 7        5  0.216688        1
    # 8        6  0.228297        1
    # 9        2  0.146017        1
    # 10       5  0.216688        1
    # 11       7  0.216047        1
    # 12       1  0.101944        1
    # 13       8  0.160711        1
    # 14      12 -0.011327       -1
    # 15      33 -0.121899       -1
    # 16       4  0.200080        1
    # 17       7  0.216047        1
    # 18       6  0.228297        1
    # 19       7  0.216047        1
    # 20       8  0.160711        1
    # 21      55 -0.199137       -1


def iris_data_od():
    '''
    内置的iris 数据集作为案例
    '''
    data = load_iris(as_frame=True)
    X, y = data.data, data.target
    df = data.frame
    print(df.head())
    # 模型训练
    iforest = IsolationForest(n_estimators=100, max_samples='auto',
                              contamination=0.05, max_features=4,
                              bootstrap=False, n_jobs=-1, random_state=1)

    #  fit_predict 函数 训练和预测一起 可以得到模型是否异常的判断，-1为异常，1为正常
    df['label'] = iforest.fit_predict(X)

    # 预测 decision_function 可以得出 异常评分
    df['scores'] = iforest.decision_function(X)

    print(df)
    #      sepal length (cm)  sepal width (cm)  ...  label    scores
    # 0                  5.1               3.5  ...      1  0.177972
    # 1                  4.9               3.0  ...      1  0.148945
    # 2                  4.7               3.2  ...      1  0.129540
    # 3                  4.6               3.1  ...      1  0.119440
    # 4                  5.0               3.6  ...      1  0.169537
    # ..                 ...               ...  ...    ...       ...
    # 145                6.7               3.0  ...      1  0.131967
    # 146                6.3               2.5  ...      1  0.122848
    # 147                6.5               3.0  ...      1  0.160523
    # 148                6.2               3.4  ...      1  0.073536
    # 149                5.9               3.0  ...      1  0.169074
    # 看看哪些预测为异常的
    print(df[df.label == -1])
    #      sepal length (cm)  sepal width (cm)  ...  label    scores
    # 13                 4.3               3.0  ...     -1 -0.039104
    # 15                 5.7               4.4  ...     -1 -0.003895
    # 41                 4.5               2.3  ...     -1 -0.038639
    # 60                 5.0               2.0  ...     -1 -0.008813
    # 109                7.2               3.6  ...     -1 -0.037663
    # 117                7.7               3.8  ...     -1 -0.046873
    # 118                7.7               2.6  ...     -1 -0.055233
    # 131                7.9               3.8  ...     -1 -0.064742
    #
    # [8 rows x 7 columns]


if __name__ == '__main__':
    created_data_od()
    iris_data_od()
