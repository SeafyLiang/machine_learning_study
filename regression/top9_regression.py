#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   top9_regression.py    
@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2022/3/15 15:17   SeafyLiang   1.0       九种顶流回归算法及实例总结
"""
from vega_datasets import data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""
资料来源：https://mp.weixin.qq.com/s/AzLWvML4KOtLDIfNNa29yg
1. 线性回归
  a. LinearRegression
2. 多项式回归
  a. PolynomialFeatures
3. 支持向量回归
  a. SVR
4. 决策树回归
  a. DecisionTreeRegressor
5. 随机森林回归
  a. RandomForestRegressor
6. LASSO回归
  a. LassoCV
7. 岭回归
  a. RidgeCV
8. ElasticNet回归
  a. ElasticNetCV
9. XGBoost回归
  a. XGBRegressor
"""


# 数据获取
df = data.cars()
print(df.head())

# 数据处理
# 过滤特定列中的NaN行
df.dropna(subset=['Horsepower', 'Miles_per_Gallon'], inplace=True)
df.sort_values(by='Horsepower', inplace=True)
# 数据转换
X = df['Horsepower'].to_numpy().reshape(-1, 1)
y = df['Miles_per_Gallon'].to_numpy().reshape(-1, 1)
plt.scatter(X, y, color='teal', edgecolors='black', label='Horsepower vs. Miles_per_Gallon')
plt.legend()
plt.show()

# 1、线性回归
from sklearn.linear_model import LinearRegression  # 创建和训练模型

linear_regressor = LinearRegression()
linear_regressor.fit(X, y)
print(linear_regressor.coef_)
# 为训练数据绘制点和拟合线
plt.scatter(X, y, color='RoyalBlue', edgecolors='black', label='Horsepower vs. Miles_per_Gallon')
plt.plot(X, linear_regressor.predict(X), color='orange', label='Linear regressor')
plt.title('Linear Regression')
plt.legend()
plt.show()

# 2、多项式回归
from sklearn.preprocessing import PolynomialFeatures

# 为二次模型生成矩阵
# 这里只是简单地生成X^0 X^1和X^2的矩阵
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
# 多项式回归模型
poly_reg_model = LinearRegression()
poly_reg_model.fit(X_poly, y)
# 为训练数据绘制点和拟合线
plt.scatter(X, y, color='DarkTurquoise', edgecolors='black',
            label='Horsepower vs. Miles_per_Gallon')
plt.plot(X, poly_reg_model.predict(X_poly), color='orange',
         label='Polynmial regressor')
plt.title('Polynomial Regression')
plt.legend()
plt.show()

# 3、支持向量回归
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler  # 执行特征缩放

scaled_X = StandardScaler()
scaled_y = StandardScaler()

scaled_X = scaled_X.fit_transform(X)
scaled_y = scaled_y.fit_transform(y)
svr_regressor = SVR(kernel='rbf', gamma='auto')
svr_regressor.fit(scaled_X, scaled_y.ravel())
plt.scatter(scaled_X, scaled_y, color='DarkTurquoise',
            edgecolors='black', label='Train')
plt.plot(scaled_X, svr_regressor.predict(scaled_X),
         color='orange', label='SVR')
plt.title('Simple Vector Regression')
plt.legend()
plt.show()

# 4、决策树回归
from sklearn.tree import DecisionTreeRegressor

# 不需要进行特性缩放，因为它将自己处理。
tree_regressor = DecisionTreeRegressor(random_state=0)
tree_regressor.fit(X, y)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(X, y, color='DarkTurquoise',
            edgecolors='black', label='Train')
plt.plot(X_grid, tree_regressor.predict(X_grid),
         color='orange', label='Tree regressor')
plt.title('Tree Regression')
plt.legend()
plt.show()

# 5、随机森林回归
from sklearn.ensemble import RandomForestRegressor

forest_regressor = RandomForestRegressor(
    n_estimators=300,
    random_state=0
)
forest_regressor.fit(X, y.ravel())
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(X, y, color='DarkTurquoise',
            edgecolors='black', label='Train')
plt.plot(X_grid, forest_regressor.predict(X_grid),
         color='orange', label='Random Forest regressor')
plt.title('Random Forest Regression')
plt.legend()
plt.show()

# 6、lasso回归
from sklearn.linear_model import LassoCV

lasso = LassoCV()
lasso.fit(X, y.ravel())
plt.scatter(X, y, color='teal', edgecolors='black',
            label='Actual observation points')
plt.plot(X, lasso.predict(X), color='orange',
         label='LASSO regressor')
plt.title('LASSO Regression')
plt.legend()
plt.show()

# 7、岭回归
from sklearn.linear_model import RidgeCV

ridge = RidgeCV()
ridge.fit(X, y)
plt.scatter(X, y, color='teal', edgecolors='black',
            label='Train')
plt.plot(X, ridge.predict(X), color='orange',
         label='Ridge regressor')
plt.title('Ridge Regression')
plt.legend()
plt.show()

# 8、elasticnet回归
from sklearn.linear_model import ElasticNetCV

elasticNet = ElasticNetCV()
elasticNet.fit(X, y.ravel())
plt.scatter(X, y, color='DarkTurquoise', edgecolors='black', label='Train')
plt.plot(X, elasticNet.predict(X), color='orange', label='ElasticNet regressor')
plt.title('ElasticNet Regression')
plt.legend()
plt.show()

# 9、xgboost回归
from xgboost import XGBRegressor

# create an xgboost regression model
model = XGBRegressor(
    n_estimators=1000,
    max_depth=7,
    eta=0.1,
    subsample=0.7,
    colsample_bytree=0.8,
)
model.fit(X, y)
plt.scatter(X, y, color='DarkTurquoise', edgecolors='black', label='Train')
plt.plot(X, model.predict(X), color='orange', label='XGBoost regressor')
plt.title('XGBoost Regression')
plt.legend()
plt.show()
