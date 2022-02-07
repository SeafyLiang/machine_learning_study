#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   HPO_algorithm.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/2/7 22:22   SeafyLiang   1.0        3个常用的超参优化算法
网格搜索（GridSearch），随机搜索（RandomSearch），贝叶斯优化（BO）
"""

import numpy as np
import pandas as pd
from lightgbm.sklearn import LGBMRegressor
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
import timeit
import os
import psutil

# 在sklearn.datasets的糖尿病数据集上演示和比较不同的算法，加载它。
diabetes = load_diabetes()
data = diabetes.data
targets = diabetes.target
n = data.shape[0]
random_state = 42
# 时间占用 s
start = timeit.default_timer()
# 内存占用 mb
info_start = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
train_data, test_data, train_targets, test_targets = train_test_split(data, targets,
                                                                      test_size=0.20, shuffle=True,
                                                                      random_state=random_state)
num_folds = 2
kf = KFold(n_splits=num_folds, random_state=random_state, shuffle=True)
model = LGBMRegressor(random_state=random_state)
score = -cross_val_score(model, train_data, train_targets, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1).mean()
print('score：', score)

end = timeit.default_timer()
info_end = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

print('此程序运行占内存' + str(info_end - info_start) + 'mB')
print('Running time:%.5fs' % (end - start))


# 1、网格搜索（GridSearch）
def grid_search():
    from sklearn.model_selection import GridSearchCV
    # 时间占用 s
    start = timeit.default_timer()
    # 内存占用 mb
    info_start = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

    param_grid = {'learning_rate': np.logspace(-3, -1, 3),
                  'max_depth': np.linspace(5, 12, 8, dtype=int),
                  'n_estimators': np.linspace(800, 1200, 5, dtype=int),
                  'random_state': [random_state]}
    gs = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error',
                      n_jobs=-1, cv=kf, verbose=False)
    gs.fit(train_data, train_targets)
    gs_test_score = mean_squared_error(test_targets, gs.predict(test_data))

    end = timeit.default_timer()
    info_end = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

    print('此程序运行占内存' + str(info_end - info_start) + 'mB')
    print('Running time:%.5fs' % (end - start))
    print("Best MSE {:.3f} params {}".format(-gs.best_score_, gs.best_params_))

    # 可视化
    import matplotlib.pyplot as plt
    gs_results_df = pd.DataFrame(np.transpose([-gs.cv_results_['mean_test_score'],
                                               gs.cv_results_['param_learning_rate'].data,
                                               gs.cv_results_['param_max_depth'].data,
                                               gs.cv_results_['param_n_estimators'].data]),
                                 columns=['score', 'learning_rate', 'max_depth',
                                          'n_estimators'])
    gs_results_df.plot(subplots=True, figsize=(10, 10))
    plt.show()


# 2、随机搜索（RandomSearch）
def random_search():
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint

    param_grid_rand = {'learning_rate': np.logspace(-5, 0, 100),
                       'max_depth': randint(2, 20),
                       'n_estimators': randint(100, 2000),
                       'random_state': [random_state]}
    # 时间占用 s
    start = timeit.default_timer()
    # 内存占用 mb
    info_start = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    rs = RandomizedSearchCV(model, param_grid_rand, n_iter=50, scoring='neg_mean_squared_error',
                            n_jobs=-1, cv=kf, verbose=False, random_state=random_state)

    rs.fit(train_data, train_targets)

    rs_test_score = mean_squared_error(test_targets, rs.predict(test_data))
    end = timeit.default_timer()
    info_end = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

    print('此程序运行占内存' + str(info_end - info_start) + 'mB')
    print('Running time:%.5fs' % (end - start))
    print("Best MSE {:.3f} params {}".format(-rs.best_score_, rs.best_params_))


# 3、贝叶斯优化（BO）
def bo_search():
    from hyperopt import fmin, tpe, hp, anneal, Trials
    def gb_mse_cv(params, random_state=random_state, cv=kf, X=train_data, y=train_targets):
        # the function gets a set of variable parameters in "param"
        params = {'n_estimators': int(params['n_estimators']),
                  'max_depth': int(params['max_depth']),
                  'learning_rate': params['learning_rate']}

        # we use this params to create a new LGBM Regressor
        model = LGBMRegressor(random_state=random_state, **params)

        # and then conduct the cross validation with the same folds as before
        score = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1).mean()

        return score

    # 状态空间，最小化函数的params的取值范围
    space = {'n_estimators': hp.quniform('n_estimators', 100, 2000, 1),
             'max_depth': hp.quniform('max_depth', 2, 20, 1),
             'learning_rate': hp.loguniform('learning_rate', -5, 0)
             }

    # trials 会记录一些信息
    trials = Trials()
    # 时间占用 s
    start = timeit.default_timer()
    # 内存占用 mb
    info_start = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    best = fmin(fn=gb_mse_cv,  # function to optimize
                space=space,
                algo=tpe.suggest,  # optimization algorithm, hyperotp will select its parameters automatically
                max_evals=50,  # maximum number of iterations
                trials=trials,  # logging
                rstate=np.random.RandomState(random_state)  # fixing random state for the reproducibility
                )

    # computing the score on the test set
    model = LGBMRegressor(random_state=random_state, n_estimators=int(best['n_estimators']),
                          max_depth=int(best['max_depth']), learning_rate=best['learning_rate'])
    model.fit(train_data, train_targets)
    tpe_test_score = mean_squared_error(test_targets, model.predict(test_data))
    end = timeit.default_timer()
    info_end = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

    print('此程序运行占内存' + str(info_end - info_start) + 'mB')
    print('Running time:%.5fs' % (end - start))
    print("Best MSE {:.3f} params {}".format(gb_mse_cv(best), best))


if __name__ == '__main__':
    grid_search()
    random_search()
    # bo_search()
