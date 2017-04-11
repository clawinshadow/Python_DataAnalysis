import numpy as np
import sklearn.preprocessing as sp

'''
some common functions would be used through data mining and machine learning
'''

def file2matrix(filename, colNo):
    with open(filename, 'r', encoding='utf-8') as f:
        arrayOLines = f.readlines()
        linesNumber = len(arrayOLines)
        returnMat = np.zeros((linesNumber, colNo))
        labels = []
        index = 0
        for line in arrayOLines:
            line = line.strip()
            data = line.split('\t')
            returnMat[index,:] = data[0:colNo]
            labels.append(data[-1])
            index += 1
    return returnMat, labels

def autoNorm(dataSet):
    '''
    将数据集标准化到[0, 1]区间内，对每列特征值的所有观测值，取最大值max
    最小值min, 每个观测值current, 格式化后的新值:
            current - min
    new = -----------------
              max - min
              
    比如在计算距离的大小时，如果每个特征值的单位度量不一样，则单位度量非常大的
    特征值将在距离中占非常大的比重，导致距离的意义失真，所以需要将每个特征值的
    数据都格式为统一的度量

    其实可以使用sklearn.preprocessing.MinMaxScaler(), 非常方便
    '''
    minVals = np.min(dataSet, axis=0)
    maxVals = np.max(dataSet, axis=0)
    ranges = maxVals - minVals
    normedDS = np.zeros(np.shape(dataSet))
    reps = dataSet.shape[0]
    normedDS = dataSet - np.tile(minVals, (reps, 1))
    normedDS = normedDS / np.tile(ranges, (reps, 1))
    return normedDS

def test():
    filepath = r'C:\Users\fan\PycharmProjects\DataAnalysis\datingTestSet2.txt'
    mat, labels = file2matrix(filepath, 3)
    print(mat)
    print(labels[0:20])
    print(mat[0:20])

    data = np.array([[1, -1, 2],
                     [2, 0, 0],
                     [0, 1, -1]])
    print('data: \n', data)
    print('autoNorm(data): \n', autoNorm(data))

    min_max_scaler = sp.MinMaxScaler()
    normedData = min_max_scaler.fit_transform(data)
    print('normed by scipy: \n', normedData)
