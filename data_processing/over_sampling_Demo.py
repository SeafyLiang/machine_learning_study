#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   over_sampling_Demo.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/4/19 17:45   SeafyLiang   1.0      过采样方法demo-SMOTE，ADASYN，RandomOverSampler
"""
from collections import Counter
from sklearn.datasets import make_classification
# pip3 install imbalanced-learn
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler

# doctest: +NORMALIZE_WHITESPACE
X, y = make_classification(
    n_classes=2,
    class_sep=2,
    weights=[0.1, 0.9],
    n_informative=3,
    n_redundant=1, flip_y=0,
    n_features=20,
    n_clusters_per_class=1,
    n_samples=1000,
    random_state=10)
print('Original dataset shape %s' % Counter(y))

# 1、SMOTE: 对于少数类样本a, 随机选择一个最近邻的样本b, 然后从a与b的连线上随机选取一个点c作为新的少数类样本。
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))

# 2、ADASYN: 关注的是在那些基于K最近邻分类器被错误分类的原始样本附近生成新的少数类样本。
ada = ADASYN(random_state=42)
X_res, y_res = ada.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))

# 3、RandomOverSampler随机采样增加新样本
ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_resampled))
