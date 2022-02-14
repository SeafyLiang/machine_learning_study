# # 第四章 逻辑回归模型 - 股票客户流失预警模型
# # 1.案例实战 - 股票客户流失预警模型
# 1.读取数据
import pandas as pd
df = pd.read_excel('股票客户流失.xlsx')
df.head()

# 2.划分特征变量和目标变量
X = df.drop(columns='是否流失') 
y = df['是否流失']   

# 3.划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train.head()
y_train.head()
X_test.head()
y_test.head()

# 4.模型搭建
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

# 5.模型使用1 - 预测数据结果
y_pred = model.predict(X_test)
print(y_pred[0:100])

a = pd.DataFrame()
a['预测值'] = list(y_pred)
a['实际值'] = list(y_test)
a.head()

from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)
print(score)

model.score(X_test, y_test)

# 6.模型使用2 - 预测概率
y_pred_proba = model.predict_proba(X_test)  
print(y_pred_proba[0:5])

a = pd.DataFrame(y_pred_proba, columns=['不流失概率', '流失概率'])
a.head()

print(y_pred_proba[:,1])

# 7.查看各个特征变量的系数（额外知识点，供参考）
print(model.coef_)
print(model.intercept_)

import numpy as np
for i in range(5):
    print(1 / (1 + np.exp(-(np.dot(X_test.iloc[i], model.coef_.T) + model.intercept_))))

# 1.读取数据
import pandas as pd
df = pd.read_excel('股票客户流失.xlsx')

# 2.划分特征变量和目标变量
X = df.drop(columns='是否流失') 
y = df['是否流失']

# 3.划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 4.模型搭建
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

# 5.模型使用1 - 预测数据结果
y_pred = model.predict(X_test)
print(y_pred[0:100])

from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)
print(score)

# 6.模型使用2 - 预测概率
y_pred_proba = model.predict_proba(X_test)  
print(y_pred_proba[0:5])

