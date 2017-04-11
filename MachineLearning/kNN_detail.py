from kNN_simple import classify
from utility import *
from sklearn.neighbors import KNeighborsClassifier

def datingClassTest():
    hoRatio = 0.1  # 划分训练集和测试集的比率
    # 导入数据
    filename = 'datingTestSet.txt'
    datingDataMat, datingLabels = file2matrix(filename, 3)
    normMat = autoNorm(datingDataMat)
    m = normMat.shape[0]
    testNumber = int(m * hoRatio)        # 取十分之一的数据为测试集
    errorCount = 0
    trainingData = normMat[testNumber:m] # 训练集
    labels = datingLabels[testNumber:m]  # 训练集的标签

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(trainingData, labels)
    for i in range(testNumber):
        # classifiedResult = classify(normMat[i], trainingData, labels, 3)
        inX = np.array(normMat[i])[np.newaxis, :]
        classifiedResult = neigh.predict(inX)
        print('the classifier came back with: {0}, the real answer is {1}'\
              .format(classifiedResult, labels[i]))
        if classifiedResult != labels[i]:
            errorCount += 1

    print('the total error rate is: {0}'.format(errorCount / testNumber))
    print(m, testNumber)

datingClassTest()
