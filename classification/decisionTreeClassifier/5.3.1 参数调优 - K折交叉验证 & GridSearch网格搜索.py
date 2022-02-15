# # 5.3 参数调优 - K折交叉验证 & GridSearch网格搜索
# **前情提要 - 5.2节的模型搭建代码**
# 1.读取数据与简单预处理
import pandas as pd
df = pd.read_excel('员工离职预测模型.xlsx')
df = df.replace({'工资': {'低': 0, '中': 1, '高': 2}})

# 2.提取特征变量和目标变量
X = df.drop(columns='离职')
y = df['离职']

# 3.划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# 4.模型训练及搭建
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=3, random_state=123)
model.fit(X_train, y_train)

# **5.3.1 K折交叉验证**
from sklearn.model_selection import cross_val_score
acc = cross_val_score(model, X, y, cv=5)
print(acc)

print(acc.mean())

from sklearn.model_selection import cross_val_score
acc = cross_val_score(model, X, y, scoring='roc_auc', cv=5)
print(acc)

print(acc.mean())

# **5.3.2 GridSearch网格搜索**
# **1.单参数的参数调优**
from sklearn.model_selection import GridSearchCV

parameters = {'max_depth': [3, 5, 7, 9, 11]}
model = DecisionTreeClassifier()

grid_search = GridSearchCV(model, parameters, scoring='roc_auc', cv=5)
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

# **补充知识点：批量生成调参所需数据**
import numpy as np
parameters = {'max_depth': np.arange(1, 10, 2)}
