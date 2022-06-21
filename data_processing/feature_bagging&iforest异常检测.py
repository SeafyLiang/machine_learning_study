#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   feature_bagging&iforest异常检测.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/6/21 11:14   SeafyLiang   1.0      FeatureBagging&IForest异常检测
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager
from pyod.models.feature_bagging import FeatureBagging  # 导入feature bagging模块
from pyod.models.iforest import IForest
# 导入isolation forest模块
from pyod.models.lof import LOF

from pyod.utils.data import generate_data, get_outliers_inliers

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

random_state = np.random.RandomState(42)
# Define seven outlier detection tools to be compared
classifiers = {
    'Feature Bagging': FeatureBagging(LOF(n_neighbors=35), contamination=outlier_fraction, check_estimator=False,
                                      random_state=random_state),
    'Isolation Forest': IForest(contamination=outlier_fraction, random_state=random_state),
}
# 设置包括feature bagging和isolation forests的分类器

# set the figure size
plt.figure(figsize=(10, 10))
# 分别用feature bagging和isolation forests进行异常值检测
for i, (clf_name, clf) in enumerate(classifiers.items()):
    # fit the dataset to the model
    clf.fit(X_train)

    # predict raw anomaly score
    scores_pred = clf.decision_function(X_train) * -1

    # prediction of a datapoint category outlier or inlier
    y_pred = clf.predict(X_train)

    # no of errors in prediction
    n_errors = (y_pred != Y_train).sum()
    print('No of Errors : ', clf_name, n_errors)

    # rest of the code is to create the visualization

    # threshold value to consider a datapoint inlier or outlier
    threshold = stats.scoreatpercentile(scores_pred, 100 * outlier_fraction)

    # decision function calculates the raw anomaly score for every point
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
    Z = Z.reshape(xx.shape)

    subplot = plt.subplot(1, 2, i + 1)

    # fill blue colormap from minimum anomaly score to threshold value
    subplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 10), cmap=plt.cm.Blues_r)

    # draw red contour line where anomaly score is equal to threshold
    a = subplot.contour(xx, yy, Z, levels=[threshold], linewidths=2, colors='red')

    # fill orange contour lines where range of anomaly score is from threshold to maximum anomaly score
    subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()], colors='orange')

    # scatter plot of inliers with white dots
    b = subplot.scatter(X_train[:-n_outliers, 0], X_train[:-n_outliers, 1], c='white', s=20, edgecolor='k')
    # scatter plot of outliers with black dots
    c = subplot.scatter(X_train[-n_outliers:, 0], X_train[-n_outliers:, 1], c='black', s=20, edgecolor='k')
    subplot.axis('tight')

    subplot.legend(
        [a.collections[0], b, c],
        ['learned decision function', 'true inliers', 'true outliers'],
        prop=matplotlib.font_manager.FontProperties(size=10),
        loc='lower right')

    subplot.set_title(clf_name)
    subplot.set_xlim((-10, 10))
    subplot.set_ylim((-10, 10))
plt.show()