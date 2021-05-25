# @File : decision_tree.py
# @Author : SeafyLiang
# @Date : 2021/5/25 10:49
# @Description : 决策树


import operator
from math import log


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVex in dataSet:
        currentLable = featVex[-1]
        if currentLable not in labelCounts.keys():
            labelCounts[currentLable] = 0
        labelCounts[currentLable] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, index, value):
    retDataset = []
    for featVec in dataSet:  # 整个样本
        if featVec[index] == value:
            reducedFeatVec = featVec[:index]  # 特征1，特征2，特征3，特征4 -> featVec[:index]  = 特征1
            reducedFeatVec.extend(featVec[index + 1:])  # featVec[index+1:] = 特征3，特征4
            retDataset.append(reducedFeatVec)
    return retDataset


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain, bestFeature = 0.0, -1
    for i in range(numFeatures):  # 色泽，声音，纹理。。。
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:  # 青绿，浅白。。。
            subdataset = splitDataSet(dataSet, i, value)
            prob = len(subdataset) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subdataset)
        bestInfoGain_ = baseEntropy - newEntropy
        if (bestInfoGain_ > bestInfoGain):
            bestInfoGain = bestInfoGain_
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # 如果数据里只有一种类别，直接返回
    # a = dataSet[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)  # 如果只有一个特征

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabels = labels[bestFeat]  # '纹理' 知道第一个特征选择的是纹理
    myTree = {bestFeatLabels: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueValus = set(featValues)
    for featValues in uniqueValus:  # 在子数据集里递归建立新的决策树
        subLabels = labels[:]
        myTree[bestFeatLabels][featValues] = createTree(splitDataSet(dataSet, bestFeat, featValues), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLable = classify(valueOfFeat, featLabels, testVec)
    else:
        classLable = valueOfFeat
    return classLable


def fishTest():
    myDat, labels = createDataSet()
    import copy
    myTree = createTree(myDat, copy.deepcopy(labels))
    print(classify(myTree, labels, [1, 1]))


if __name__ == "__main__":
    fishTest()
