# # 4.1 逻辑回归模型算法原理
# **4.1.1 逻辑回归模型的数学原理**
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-6, 6)
y = 1.0 / (1.0 + np.exp(-x))
plt.plot(x,y)
plt.show()

import numpy as np
x = np.linspace(-6, 6)
print(x)

x = -1
np.exp(-x)

# **4.1.2 逻辑回归模型的代码实现**
X = [[1, 0], [5, 1], [6, 4], [4, 2], [3, 2]]
y = [0, 1, 1, 0, 0]

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, y)  

import warnings
warnings.filterwarnings('ignore')

print(model.predict([[2,2]]))

print(model.predict([[1,1], [2,2], [5, 5]]))

print(model.predict([[1, 0], [5, 1], [6, 4], [4, 2], [3, 2]]))  # 因为这里演示的多个数据和X是一样的，所以也可以直接写成model.predict(X)

# **4.1.3 逻辑回归模型的深入理解**
y_pred_proba = model.predict_proba(X)
print(y_pred_proba)

import pandas as pd
a = pd.DataFrame(y_pred_proba, columns=['分类为0的概率', '分类为1的概率'])  # 2.2.1 通过numpy数组创建DataFrame
print(a)

print(model.coef_)
print(model.intercept_)

model.coef_.T

import numpy as np
for i in range(5):
    print(1 / (1 + np.exp(-(np.dot(X[i], model.coef_.T) + model.intercept_))))

X = [[1, 0], [5, 1], [6, 4], [4, 2], [3, 2]]
y = [-1, 0, 1, 1, 1]

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, y)

print(model.predict([[0, 0]]))

model.predict(X)

print(model.predict_proba([[0, 0]]))

