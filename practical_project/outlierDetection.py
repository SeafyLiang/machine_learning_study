#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   outlierDetection.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2021/8/12 14:06   SeafyLiang   1.0       离群点检测pyod
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager
# 检测数据集中异常值的模型
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
# from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
# from pyod.models.loci import LOCI
# from pyod.models.sod import SOD
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.utils.data import generate_data, get_outliers_inliers

"""
参考资料：
https://blog.csdn.net/weixin_41697507/article/details/89408236
https://blog.csdn.net/sparkexpert/article/details/81195418
https://github.com/yzhao062/Pyod#ramaswamy2000efficient
"""

# 创建一个带有异常值的随机数据集并绘制它
# generate random data with two features
X_train, Y_train = generate_data(n_train=200, train_only=True, n_features=2)
# by default the outlier fraction is 0.1 in generate data function
outlier_fraction = 0.1
# store outliers and inliers in different numpy arrays
x_outliers, x_inliers = get_outliers_inliers(X_train, Y_train)
n_inliers = len(x_inliers)
n_outliers = len(x_outliers)
# separate the two features and use it to plot the data
F1 = X_train[:, [0]].reshape(-1, 1)
F2 = X_train[:, [1]].reshape(-1, 1)
# create a meshgrid
xx, yy = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200))
# scatter plot
plt.scatter(F1, F2)
plt.xlabel('F1')
plt.ylabel('F2')
plt.show()

"""
Model 1 Angle-based Outlier Detector (ABOD)
Model 2 Cluster-based Local Outlier Factor (CBLOF)
Model 3 Feature Bagging
Model 4 Histogram-base Outlier Detection (HBOS)
Model 5 Isolation Forest
Model 6 K Nearest Neighbors (KNN)
Model 7 Fast outlier detection using the local correlation integral(LOCI)
Model 8 Subspace Outlier Detection (SOD)
Model 9 Local Outlier Factor (LOF)
Model 10 Minimum Covariance Determinant (MCD)
Model 11 One-class SVM (OCSVM)
Model 12 Principal Component Analysis (PCA)
"""
# 创建一个dictionary并添加要用于检测异常值的所有模型
classifiers = {
    'ABOD': ABOD(contamination=outlier_fraction),
    'CBLOF': CBLOF(contamination=outlier_fraction),
    # 'Feature Bagging': FeatureBagging(contamination=outlier_fraction),
    'HBOS': HBOS(contamination=outlier_fraction),
    'IForest': IForest(contamination=outlier_fraction),
    'KNN': KNN(contamination=outlier_fraction),
    # 'LOCI': LOCI(contamination=outlier_fraction, ),
    # 'SOD': SOD(contamination=outlier_fraction, ),
    'LOF': LOF(contamination=outlier_fraction, ),
    'MCD': MCD(contamination=outlier_fraction, ),
    'OCSVM': OCSVM(contamination=outlier_fraction, ),
    'PCA': PCA(contamination=outlier_fraction, ),

}
# 将数据拟合到我们在dictionary中添加的每个模型，然后，查看每个模型如何检测异常值
# set the figure size

plt.figure(figsize=(10, 10))
for i, (clf_name, clf) in enumerate(classifiers.items()):
    print()
    print(i + 1, 'fitting', clf_name)
    # fit the data and tag outliers
    clf.fit(X_train)
    scores_pred = clf.decision_function(X_train) * -1
    y_pred = clf.predict(X_train)
    threshold = stats.scoreatpercentile(scores_pred,
                                        100 * outlier_fraction)
    n_errors = (y_pred != Y_train).sum()
    # plot the levels lines and the points

    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
    Z = Z.reshape(xx.shape)
    subplot = plt.subplot(3, 4, i + 1)
    subplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),
                     cmap=plt.cm.Blues_r)
    a = subplot.contour(xx, yy, Z, levels=[threshold],
                        linewidths=2, colors='red')
    subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()],
                     colors='orange')
    b = subplot.scatter(X_train[:-n_outliers, 0], X_train[:-n_outliers, 1], c='white',
                        s=20, edgecolor='k')
    c = subplot.scatter(X_train[-n_outliers:, 0], X_train[-n_outliers:, 1], c='black',
                        s=20, edgecolor='k')
    subplot.axis('tight')
    subplot.legend(
        [a.collections[0], b, c],
        ['learned decision function', 'true inliers', 'true outliers'],
        prop=matplotlib.font_manager.FontProperties(size=10),
        loc='lower right')
    subplot.set_xlabel("%d. %s (errors: %d)" % (i + 1, clf_name, n_errors))
    subplot.set_xlim((-7, 7))
    subplot.set_ylim((-7, 7))
plt.show()
