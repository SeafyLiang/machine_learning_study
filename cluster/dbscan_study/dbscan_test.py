#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   dbscan_test.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/2/9 23:08   SeafyLiang   1.0         dbscan聚类算法demo，与kmeans对比
"""
# 1.读取数据
import pandas as pd

data = pd.read_excel('演示数据.xlsx')
data.head()

# 2.数据可视化
import matplotlib.pyplot as plt

plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c="green", marker='*')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 3.数据建模
from sklearn.cluster import DBSCAN

dbs = DBSCAN()
dbs.fit(data)
label_dbs = dbs.labels_

# 4.查看聚类结果
print(label_dbs)

# 5.用散点图展示DBSCAN算法的聚类结果
plt.scatter(data[label_dbs == 0].iloc[:, 0], data[label_dbs == 0].iloc[:, 1], c="red", marker='o', label='class0')
plt.scatter(data[label_dbs == 1].iloc[:, 0], data[label_dbs == 1].iloc[:, 1], c="green", marker='*', label='class1')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# # 13.2.3 KMeans VS DBSCAN
from sklearn.cluster import KMeans

KMs = KMeans(n_clusters=2)
KMs.fit(data)
label_kms = KMs.labels_
print(label_kms)

plt.scatter(data[label_kms == 0].iloc[:, 0], data[label_kms == 0].iloc[:, 1], c="red", marker='o', label='class0')
plt.scatter(data[label_kms == 1].iloc[:, 0], data[label_kms == 1].iloc[:, 1], c="green", marker='*', label='class1')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
