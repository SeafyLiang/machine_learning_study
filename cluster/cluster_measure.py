#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   cluster_measure.py
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/2/23 23:17   SeafyLiang   1.0      3种聚类算法内部度量-si,ch,dbi
"""
import pandas as pd
from numpy import unique
from numpy import where
from matplotlib import pyplot
from sklearn import metrics
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

data = pd.read_excel('cluster_data.xlsx')
eps_list = []
si_list = []
ch_list = []
dbi_list = []

"""
聚类-内部度量
Calinski-Harabaz Index：

在scikit-learn中， Calinski-Harabasz Index对应的方法是metrics.calinski_harabaz_score.
CH指标通过计算类中各点与类中心的距离平方和来度量类内的紧密度，通过计算各类中心点与数据集中心点距离平方和来度量数据集的分离度，CH指标由分离度与紧密度的比值得到。从而，CH越大代表着类自身越紧密，类与类之间越分散，即更优的聚类结果。

sklearn.metrics.silhouette_score:轮廓系数

silhouette_sample
对于一个样本点(b - a)/max(a, b)
a平均类内距离，b样本点到与其最近的非此类的距离。
silihouette_score返回的是所有样本的该值,取值范围为[-1,1]。

这些度量均是越大越好
"""
# 初始参数聚类
model = DBSCAN(min_samples=2)
# 模型拟合与聚类预测
X = data[['x', 'y']].values
yhat = model.fit_predict(X)
# 检索唯一群集
clusters = unique(yhat)
# 为每个群集的样本创建散点图
for cluster in clusters:
    # 获取此群集的示例的行索引
    row_ix = where(yhat == cluster)
    # 创建这些样本的散布
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=cluster)
# 绘制散点图
pyplot.xlabel('x')
pyplot.ylabel('y')
plt.legend()
pyplot.show()

eps_cal = 1
while eps_cal < 30:
    model = DBSCAN(eps=eps_cal, min_samples=2)
    label_list = model.fit_predict(data)
    # 轮廓系数
    cluster_score_si = metrics.silhouette_score(data, label_list)

    cluster_score_ch = metrics.calinski_harabasz_score(data, label_list)

    # DBI的值最小是0，值越小，代表聚类效果越好。
    cluster_score_DBI = metrics.davies_bouldin_score(data, label_list)

    eps_list.append(eps_cal)
    si_list.append(cluster_score_si)
    ch_list.append(cluster_score_ch)
    dbi_list.append(cluster_score_DBI)
    eps_cal += 1

plt.figure()
plt.plot(eps_list, si_list)
plt.xlabel("dbscan-eps")
plt.ylabel("silhouette_score")
plt.title("dbscan-eps-si")
plt.show()

plt.figure()
plt.plot(eps_list, ch_list)
plt.xlabel("dbscan-eps")
plt.ylabel("calinski_harabasz_score")
plt.title("dbscan-eps-ch")
plt.show()

plt.figure()
plt.plot(eps_list, dbi_list)
plt.xlabel("dbscan-eps")
plt.ylabel("davies_bouldin_score")
plt.title("dbscan-eps-dbi")
plt.show()

# eps经过参数调优，选择22为合适值
model = DBSCAN(eps=22, min_samples=2)
# 模型拟合与聚类预测
X = data[['x', 'y']].values
yhat = model.fit_predict(X)
# 检索唯一群集
clusters = unique(yhat)
# 为每个群集的样本创建散点图
for cluster in clusters:
    # 获取此群集的示例的行索引
    row_ix = where(yhat == cluster)
    # 创建这些样本的散布
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=cluster)
# 绘制散点图
pyplot.xlabel('x')
pyplot.ylabel('y')
plt.legend()
pyplot.show()
