# @File : 02logical_regression.py.py
# @Author : SeafyLiang
# @Date : 2021/4/17 23:35
# @Description : 逻辑回归

from numpy import *
import matplotlib.pylab as plt


# 解析数据
def loadDataSet(file_name):
    dataMat = []
    labelMat = []
    fr = open(file_name)
    for line in fr.readlines():
        curLine = line.strip().split()
        dataMat.append([1.0,float(curLine[0]),float(curLine[1])])
        labelMat.append(int(curLine[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return  1.0 / (1 + exp(-inX))


def stocGradAscent1(dataMatrix, classLabels, numIter=500):
    m,n = shape(dataMatrix)
    weights = ones(n)   # 创建与列数相同的矩阵的系数矩阵，所有的元素都是1
    # 随机梯度, 循环150,观察是否收敛
    alpha = 0.001
    for j in range(numIter):
        # [0, 1, 2 .. m-1]
        dataIndex = range(m)
        for i in range(m):
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[dataIndex[randIndex]]*weights))
            error = classLabels[dataIndex[randIndex]] - h
            weights = weights + alpha * error * dataMatrix[dataIndex[randIndex]]
            # del(dataIndex[randIndex])
    return weights


def plotBestFit(dataArr, labelMat, weights):


    n = shape(dataArr)[0]
    xcord1 = [];
    ycord1 = []
    xcord2 = [];
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]);
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]);
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)

    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X');
    plt.ylabel('Y')
    plt.show()


def testLR():
    # 1.收集并准备数据
    dataMat, labelMat = loadDataSet("data/data2.txt")
    dataArr = array(dataMat)
    # print dataArr
    weights = stocGradAscent1(dataArr, labelMat)
    # 数据可视化
    plotBestFit(dataArr, labelMat, weights)


if __name__ == '__main__':
    testLR()

