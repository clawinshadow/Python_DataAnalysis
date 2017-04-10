import numpy as np
import operator
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

'''
kNN算法的简单示例
一般逻辑如下：
1. 计算已知类别数据集中的点与当前点之间的距离
2. 按照距离递增次序排序
3. 选取与当前点距离最小的k个点
4. 确定前k个点所在类别的出现频率
5. 返回前k个点出现频率最高的类别作为当前点的预测分类
'''

def createDataSet():
    group = np.array([[1, 1.1], [1, 1], [0, 0], [0, 0.1]])
    labels = np.array(['A', 'A', 'B', 'B'])
    return group, labels

def classify(inX, dataSet, labels, k):
    dsSize = dataSet.shape[0]                            # 样本数量
    diffMat = np.tile(inX, (dsSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    distances = np.power(np.sum(sqDiffMat, axis=1), 0.5) # 计算inX与各样本之间的距离
    sortedIndices = distances.argsort()                  # 返回排好序后的索引集合

    # 获取最近k个距离的每个相应的label，并计算每个label的数量，存于dict中
    labelCountDict = dict()
    for i in range(k):
        label = labels[sortedIndices[i]]
        labelCountDict[label] = labelCountDict.get(label, 0) + 1

    # 将dict按label的数量倒序排列
    sortedLabelCount = sorted(iter(labelCountDict.items()), key=operator.itemgetter(1), reverse=True)

    # 返回数量最多的label
    return sortedLabelCount[0][0]

group, labels = createDataSet()
inX = [0.2, 0.6]
print('predict inX({0}): {1}'.format(inX, classify(inX, group, labels, 3)))

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(group, labels)
inX2 = np.array(inX)[np.newaxis, :] # predict方法只接受2维数组作为参数
print('predict inX({0}) by sklearn: {1}'.format(inX, neigh.predict(inX2)))
print('predict probability of inX({0}) by sklearn: {1}'.format(inX, neigh.predict_proba(inX2)))
      
# Visualize
Apoints = []
Bpoints = []
groupSize = group.shape[0]
for i in range(groupSize):
    if labels[i] == 'A':
        Apoints.append(group[i].tolist())
    else:
        Bpoints.append(group[i].tolist())

Apoints = np.array(Apoints)
Bpoints = np.array(Bpoints)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(Apoints[:, 0], Apoints[:, 1], c='red', marker='o')
for a in Apoints:
    plt.annotate('A', xy=(a[0], a[1]))
    
ax.scatter(Bpoints[:, 0], Bpoints[:, 1], c='blue', marker='v')
for b in Bpoints:
    plt.annotate('B', xy=(b[0], b[1]))

ax.scatter(inX[0], inX[1], c='green', marker='s')
plt.axis([-1, 1.5, -1, 1.5])
plt.show()
    
