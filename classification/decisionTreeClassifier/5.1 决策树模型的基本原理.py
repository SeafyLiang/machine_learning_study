# # 5.1 决策树模型的基本原理
# # 5.1.3 决策树模型的代码实现
# **1.分类决策树模型（DecisionTreeClassifier）**
from sklearn.tree import DecisionTreeClassifier
X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
y = [1, 0, 0, 1, 1]

model = DecisionTreeClassifier(random_state=0)
model.fit(X, y)

print(model.predict([[5, 5]]))

print(model.predict([[5, 5], [7, 7], [9, 9]]))

# **补充知识点：决策树可视化（供感兴趣的读者参考）**
from sklearn.tree import export_graphviz
import graphviz

import os  # 以下两行是环境变量配置，运行一次即可，相关知识点可以参考5.2.3节
os.environ['PATH'] = os.pathsep + r'C:\Program Files (x86)\Graphviz2.38\bin'

dot_data = export_graphviz(model, out_file=None, class_names=['0', '1'])
graph = graphviz.Source(dot_data)
print(graph)

# **补充知识点：random_state参数的作用解释**
from sklearn.tree import DecisionTreeClassifier
X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
y = [1, 0, 0, 1, 1]

model = DecisionTreeClassifier()
model.fit(X, y)

dot_data = export_graphviz(model, out_file=None, class_names=['0', '1'])
graph = graphviz.Source(dot_data)
print(graph)

from sklearn.tree import DecisionTreeClassifier
X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
y = [1, 0, 0, 1, 1]

model = DecisionTreeClassifier()
model.fit(X, y)

dot_data = export_graphviz(model, out_file=None, class_names=['0', '1'])
graph = graphviz.Source(dot_data)
print(graph)

# **2.回归决策树模型（DecisionTreeRegressor）**
from sklearn.tree import DecisionTreeRegressor
X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
y = [1, 2, 3, 4, 5]

model = DecisionTreeRegressor(max_depth=2, random_state=0)
model.fit(X, y)

print(model.predict([[9, 9]]))

dot_data = export_graphviz(model, out_file=None)
graph = graphviz.Source(dot_data)
print(graph)

