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
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=3, random_state=123)
model.fit(X_train, y_train)

# **2.预测不离职&离职概率**
y_pred_proba = model.predict_proba(X_test)
print(y_pred_proba[0:5])

# **3.模型预测效果评估**
from sklearn.metrics import roc_curve
fpr, tpr, thres = roc_curve(y_test, y_pred_proba[:,1])

a = pd.DataFrame()
a['阈值'] = list(thres)
a['假警报率'] = list(fpr)
a['命中率'] = list(tpr)
a.head()

import matplotlib.pyplot as plt
plt.plot(fpr, tpr)
plt.show()

from sklearn.metrics import roc_auc_score
score = roc_auc_score(y_test, y_pred_proba[:,1])
print(score)

# **4.特征重要性评估**
print(model.feature_importances_)

features = X.columns
importances = model.feature_importances_

importances_df = pd.DataFrame()
importances_df['特征名称'] = features
importances_df['特征重要性'] = importances
importances_df.sort_values('特征重要性', ascending=False)

# **5.2.3 决策树模型可视化呈现及决策树要点理解**
# 1.如果不用显示中文，那么通过如下代码即可：
# !pip3 install pygraphviz

from sklearn.tree import export_graphviz
import graphviz
import os
os.environ['PATH'] = os.pathsep + r'C:\Program Files (x86)\Graphviz2.38\bin'

dot_data = export_graphviz(model, out_file=None, class_names=['0', '1'])
graph = graphviz.Source(dot_data)

graph.render("result")
print('可视化文件result.pdf已经保存在代码所在文件夹！')

print(graph)

dot_data = export_graphviz(model, out_file=None, feature_names=['income', 'satisfication', 'score', 'project_num', 'hours', 'year'], class_names=['0', '1'], filled=True)
graph = graphviz.Source(dot_data)

print(graph)

# 2.如果想显示中文，需要使用如下代码
from sklearn.tree import export_graphviz
import graphviz
import os  # 以下这两行是手动进行环境变量配置，防止在本机环境的变量部署失败
os.environ['PATH'] = os.pathsep + r'C:\Program Files (x86)\Graphviz2.38\bin'

dot_data = export_graphviz(model, out_file=None, feature_names=X_train.columns, class_names=['不离职', '离职'], rounded=True, filled=True)

f = open('dot_data.txt', 'w')
f.write(dot_data)
f.close()

import re
f_old = open('dot_data.txt', 'r')
f_new = open('dot_data_new.txt', 'w', encoding='utf-8')
for line in f_old:
    if 'fontname' in line:
        font_re = 'fontname=(.*?)]'
        old_font = re.findall(font_re, line)[0]
        line = line.replace(old_font, 'SimHei')
    f_new.write(line)
f_old.close()
f_new.close()

os.system('dot -Tpng dot_data_new.txt -o 决策树模型.png')
print('决策树模型.png已经保存在代码所在文件夹！')

os.system('dot -Tpdf dot_data_new.txt -o 决策树模型.pdf')
print('决策树模型.pdf已经保存在代码所在文件夹！')
