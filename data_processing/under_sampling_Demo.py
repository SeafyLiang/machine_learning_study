#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   under_sampling_Demo.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/4/19 17:55   SeafyLiang   1.0       欠采样demo-RandomUnderSampler,NearMiss
"""
from collections import Counter
from sklearn.datasets import make_classification

# doctest: +NORMALIZE_WHITESPACE
X, y = make_classification(
    n_classes=2,
    class_sep=2,
    weights=[0.1, 0.9],
    n_informative=3,
    n_redundant=1,
    flip_y=0,
    n_features=20,
    n_clusters_per_class=1,
    n_samples=1000,
    random_state=10)
print('Original dataset shape %s' % Counter(y))

# 1、RandomUnderSampler函数是一种快速并十分简单的方式来平衡各个类别的数据----随机选取数据的子集。
from imblearn.under_sampling import RandomUnderSampler

model_undersample = RandomUnderSampler()
x_undersample_resampled, y_undersample_resampled = model_undersample.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_undersample_resampled))

# 2、NearMiss函数则添加了一些启发式heuristic的规则来选择样本, 通过设定version参数来实现三种启发式的规则。
from imblearn.under_sampling import NearMiss

nm1 = NearMiss(version=1)
X_resampled_num1, y_resampled = nm1.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_resampled))
