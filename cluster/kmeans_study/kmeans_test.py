import pandas

data = pandas.read_csv(
    '电信话单.csv',
    encoding='utf8', engine='python'
)

import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# 设置中文字体
# 解决 plt 中文显示的问题 mymac
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']


fColumns = [
    '工作日上班电话时长',
    '工作日下班电话时长',
    '周末电话时长', '国际电话时长',
    '总电话时长', '平均每次通话时长'
]

plt.figure()
# 绘制散点矩阵图
axes = scatter_matrix(
    data[fColumns], diagonal='hist'
)

# 设置坐标轴的字体，避免坐标轴出现中文乱码
for ax in axes.ravel():
    ax.set_xlabel(
        ax.get_xlabel()
    )
    ax.set_ylabel(
        ax.get_ylabel()
    )

# 计算相关系数矩阵
dCorr = data[fColumns].corr()

fColumns = [
    '工作日上班电话时长', '工作日下班电话时长',
    '周末电话时长', '国际电话时长', '平均每次通话时长'
]

from sklearn.decomposition import PCA

pca_2 = PCA(n_components=2)
data_pca_2 = pandas.DataFrame(
    pca_2.fit_transform(data[fColumns])
)
plt.scatter(
    data_pca_2[0],
    data_pca_2[1]
)

from sklearn.cluster import KMeans

kmModel = KMeans(n_clusters=3)
kmModel = kmModel.fit(data[fColumns])

pTarget = kmModel.predict(data[fColumns])

plt.figure()
plt.scatter(
    data_pca_2[0],
    data_pca_2[1],
    c=pTarget
)

pandas.crosstab(pTarget, pTarget)

import seaborn as sns
from pandas.plotting import parallel_coordinates

fColumns = [
    '工作日上班电话时长',
    '工作日下班电话时长',
    '周末电话时长',
    '国际电话时长',
    '平均每次通话时长',
    '类型'
]

data['类型'] = pTarget

plt.figure()
ax = parallel_coordinates(
    data[fColumns], '类型',
    color=sns.color_palette(),
)
# 设置坐标轴的字体，避免坐标轴出现中文乱码
ax.set_xticklabels(
    ax.get_xticklabels()
)
ax.set_yticklabels(
    ax.get_yticklabels()
)
plt.show()