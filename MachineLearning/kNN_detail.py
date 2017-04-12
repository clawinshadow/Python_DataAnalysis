from kNN_simple import classify
from utility import *
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

'''
k-近邻算法是基于实例的学习，它有一些缺点：
1. 需要有接近实际数据的训练样本数据
2. 必须保存全部数据集，如果训练数据很大，则需使用大量的存储空间
3. 由于必须对数据集中的每个数据计算距离值，通常会非常耗时
4. 因为它没有分析过训练样本数据集，所以它无法给出每个类别数据的具体特征

优点是分类非常准确
'''

def datingClassTest():
    hoRatio = 0.1                                # 划分训练集和测试集的比率
    # 导入数据
    filename = 'datingTestSet.txt'
    datingDataMat, datingLabels = file2matrix(filename, 3)
    normMat = autoNorm(datingDataMat)
    m = normMat.shape[0]
    testNumber = int(m * hoRatio)                # 取十分之一的数据为测试集
    errorCount = 0
    trainingData = normMat[testNumber:m]         # 训练集
    trainingLabels = datingLabels[testNumber:m]  # 训练集的标签

    neigh = KNeighborsClassifier(n_neighbors=7)
    neigh.fit(trainingData, trainingLabels)
    for i in range(testNumber):
        # classifiedResult = classify(normMat[i], trainingData, labels, 3)
        inX = np.array(normMat[i])[np.newaxis, :]
        classifiedResult = neigh.predict(inX)
        print('the classifier came back with: {0}, the real answer is {1}'\
              .format(classifiedResult, datingLabels[i]))
        if classifiedResult != datingLabels[i]:
            errorCount += 1

    print('the total error rate is: {0}'.format(errorCount / testNumber))
    print('total error count: ', errorCount)

def classifyPerson():
    labels = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input('Percentage of time spent playing video games?'))
    ffMiles = float(input('frequent flier miles earned per year?'))
    iceCream = float(input('liters of ice cream consumed per year?'))
    filename = 'datingTestSet2.txt'
    datingDataMat, datingLabels = file2matrix(filename, 3)
    inX = np.array([ffMiles, percentTats, iceCream])

    finalMat = np.vstack((inX, datingDataMat))
    normMat = autoNorm(finalMat) # 标准化的时候需要将待分类的样本也一起加进去
    
    classifiedResult = classify(normMat[0], normMat[1:], datingLabels, 3)
    print('you''ll probably like this person: ', labels[int(classifiedResult) - 1])
    
    
datingClassTest()
classifyPerson()
