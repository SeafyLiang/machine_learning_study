# **4.3.2 案例实战 - 评估股票客户流失预警模型**
# 1.读取数据
import pandas as pd
df = pd.read_excel('股票客户流失.xlsx')

# 2.划分特征变量和目标变量
X = df.drop(columns='是否流失')
y = df['是否流失']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

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
y_pred_proba = model.predict_proba(X_test)
print(y_pred_proba[0:5])
y_pred_proba[:,1]

# 1.计算ROC曲线需要的假警报率（fpr）、命中率（tpr）及阈值（thres）
from sklearn.metrics import roc_curve
fpr, tpr, thres = roc_curve(y_test, y_pred_proba[:,1])

print(roc_curve(y_test, y_pred_proba[:,1]))
type(roc_curve(y_test, y_pred_proba[:,1]))
len(roc_curve(y_test, y_pred_proba[:,1]))

# 2.查看假警报率（fpr）、命中率（tpr）及阈值（thres）
a = pd.DataFrame()
a['阈值'] = list(thres)
a['假警报率'] = list(fpr)
a['命中率'] = list(tpr)
a.head()

# 3.绘制ROC曲线
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.plot(fpr, tpr)
plt.title('ROC曲线')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

# 4.求出模型的AUC值
from sklearn.metrics import roc_auc_score
score = roc_auc_score(y_test, y_pred_proba[:,1])
print(score)

# **补充知识点：对阈值取值的理解**
max(y_pred_proba[:,1])

a = pd.DataFrame(y_pred_proba, columns=['分类为0概率', '分类为1概率'])
a = a.sort_values('分类为1概率', ascending=False)
a.head(15)

# # 补充知识点：KS曲线绘制
from sklearn.metrics import roc_curve
fpr, tpr, thres = roc_curve(y_test, y_pred_proba[:,1])

a = pd.DataFrame()  # 创建一个空DataFrame
a['阈值'] = list(thres)
a['假警报率'] = list(fpr)
a['命中率'] = list(tpr)
a.head()

plt.plot(thres[1:], tpr[1:])
plt.plot(thres[1:], fpr[1:])
plt.plot(thres[1:], tpr[1:] - fpr[1:])
plt.xlabel('threshold')
plt.legend(['tpr', 'fpr', 'tpr-fpr'])
plt.gca().invert_xaxis() 
plt.show()

print(max(tpr - fpr))

a['TPR-FPR'] = a['命中率'] - a['假警报率']
a.head()

print(max(a['TPR-FPR']))

a[a['TPR-FPR'] == max(a['TPR-FPR'])]
