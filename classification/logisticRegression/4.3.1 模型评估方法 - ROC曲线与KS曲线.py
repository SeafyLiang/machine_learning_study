# # 4.3 模型评估方法 - ROC曲线与KS曲线
# # 1.案例实战 - 股票客户流失预警模型
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

# 查看全部的预测准确度
from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)
print(score)

# 6.模型使用2 - 预测概率
y_pred_proba = model.predict_proba(X_test)  
print(y_pred_proba[0:5])

# # 4.3 模型评估方法 - ROC曲线与KS曲线
# **4.3.1 分类模型的评估方法 - ROC曲线**
# 补充知识点：混淆矩阵的Python实现

from sklearn.metrics import confusion_matrix
m = confusion_matrix(y_test, y_pred)  # 传入预测值和真实值
print(m)

a = pd.DataFrame(m, index=['0（实际不流失）', '1（实际流失）'], columns=['0（预测不流失）', '1（预测流失）'])
print(a)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
