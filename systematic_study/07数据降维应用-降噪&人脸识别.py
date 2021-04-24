#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   07数据降维应用-降噪&人脸识别.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2021/4/24 15:10   SeafyLiang   1.0     数据降维应用（PCA）-降噪&人脸识别
"""

# 1. 降噪
import numpy as np
import matplotlib.pyplot as plt

# 造含有噪声的数据
X = np.empty((100, 2))
X[:, 0] = np.random.uniform(0., 100., size=100)
X[:, 1] = 0.75 * X[:, 0] + 3. + np.random.normal(0, 5, size=100)
plt.scatter(X[:, 0], X[:, 1])
plt.show()

# 对含噪声数据先降维，再还原，就实现了降噪
from sklearn.decomposition import PCA

pca = PCA(n_components=1)
pca.fit(X)
X_reduction = pca.transform(X)
X_restore = pca.inverse_transform(X_reduction)
plt.scatter(X_restore[:, 0], X_restore[:, 1])
plt.show()

from sklearn import datasets

digits = datasets.load_digits()
X = digits.data
y = digits.target
# 添加一个正态分布的噪音矩阵
noisy_digits = X + np.random.normal(0, 4, size=X.shape)
# 绘制噪音数据:从y==0数字中取10个,进行10次循环;
# 依次从y=num再取出10个,将其与原来的样本拼在一起
example_digits = noisy_digits[y == 0, :][:10]
for num in range(1, 10):
    example_digits = np.vstack([example_digits, noisy_digits[y == num, :][:10]])
print(example_digits.shape)  # (100, 64)


# 绘制100个数字
def plot_digits(data):
    fig, axes = plt.subplots(10, 10, figsize=(10, 10),
                             subplot_kw={'xticks': [], 'yticks': []},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8, 8),
                  cmap='binary', interpolation='nearest', clim=(0, 16))
    plt.show()


plot_digits(example_digits)

# 使用PCA降噪
pca = PCA(0.5).fit(noisy_digits)
pca.n_components_
# 输出:12,即原始数据保存了50%的信息,需要12维
# 进行降维、还原过程
components = pca.transform(example_digits)
filtered_digits = pca.inverse_transform(components)
plot_digits(filtered_digits)

# 2. 人脸识别
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people

faces = fetch_lfw_people()
print(faces.keys())
# 输出：dict_keys('data', 'images','target','target_names','DESCR')
print(faces.data.shape)
# 输出：(13233,2914)
# mages属性,即对每一维样本,将其以2维的形式展现出来
# 即对于每个样本来说都是62*47
print(faces.images.shape)
# 输出：(13233,62,47)
random_indexes = np.random.permutation(len(faces.data))
X = faces.data[random_indexes]
example_faces = X[:36, :]
print(example_faces.shape)


# 输出：(36,2914)


# 一万多张脸抽出36张脸
def plot_faces(faces):
    fig, axes = plt.subplots(6, 6, figsize=(10, 10),
                             subplot_kw={'xticks': [], 'yticks': []},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(faces[i].reshape(62, 47), cmap='bone')
        plt.show()


plot_faces(example_faces)

# 特征脸
from sklearn.decomposition import PCA

# %%time notebook
# 实例化PCA,求出所有的主成分
pca = PCA(svd_solver='randomized')
pca.fit(X)

# 输出所有的主成分(向量)以及每个主成分的特征
print(pca.components_.shape)
# 输出：(2914, 2914)

# 特征脸就是主成分中的每一行都当作一个样本,排名越靠前,表示该特征脸越能表示人脸的特征。取出前36个特征脸 进行可视化
plot_faces(pca.components_[:36, :])
