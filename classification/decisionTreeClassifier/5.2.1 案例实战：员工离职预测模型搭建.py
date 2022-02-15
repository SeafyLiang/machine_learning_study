# # 5.2 案例实战：员工离职预测模型搭建
# **5.2.1 模型搭建**
# 1.数据读取与预处理
import pandas as pd
df = pd.read_excel('员工离职预测模型.xlsx')
df.head()

df = df.replace({'工资': {'低': 0, '中': 1, '高': 2}})
df.head()

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

# **5.2.2 模型预测及评估**
# **1.直接预测是否离职**
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

# **2.预测不离职&离职概率**
y_pred_proba = model.predict_proba(X_test)
print(y_pred_proba[0:5])

b = pd.DataFrame(y_pred_proba, columns=['不离职概率', '离职概率']) 
b.head()

print(y_pred_proba[:,1])
