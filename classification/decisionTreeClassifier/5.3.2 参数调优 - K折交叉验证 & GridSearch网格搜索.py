# **2.参数调优的效果检验**
# 2.1 查看新模型准确度
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
model = DecisionTreeClassifier(max_depth=7)
model.fit(X_train, y_train) 

y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)
print(score)

# 2.2 查看新模型的ROC曲线和AUC值
y_pred_proba = model.predict_proba(X_test)
print(y_pred_proba[:,1])

from sklearn.metrics import roc_curve
fpr, tpr, thres = roc_curve(y_test, y_pred_proba[:,1])

import matplotlib.pyplot as plt
plt.plot(fpr, tpr)
plt.show()

from sklearn.metrics import roc_auc_score
score = roc_auc_score(y_test, y_pred_proba[:,1])
print(score)

print(model.feature_importances_)

features = X.columns
importances = model.feature_importances_

importances_df = pd.DataFrame()
importances_df['特征名称'] = features
importances_df['特征重要性'] = importances
importances_df.sort_values('特征重要性', ascending=False)

# **3.多参数调优**
from sklearn.model_selection import GridSearchCV
parameters = {'max_depth': [5, 7, 9, 11, 13], 'criterion':['gini', 'entropy'], 'min_samples_split':[5, 7, 9, 11, 13, 15]}
model = DecisionTreeClassifier()

grid_search = GridSearchCV(model, parameters, scoring='roc_auc', cv=5)
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

model = DecisionTreeClassifier(criterion='entropy', max_depth=11, min_samples_split=13)
model.fit(X_train, y_train) 

y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)
print(score)

y_pred_proba = model.predict_proba(X_test)
print(y_pred_proba[:,1])

score = roc_auc_score(y_test, y_pred_proba[:,1])
print(score)
