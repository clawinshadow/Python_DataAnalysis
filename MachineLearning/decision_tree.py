import math
import numpy as np
import sklearn.tree as st

'''
决策树的中心思想在于每次划分都要最大程度上提高数据集的纯度(purity)，如何衡量数据集的纯度主要有三种方法：
令 数据集为D, |D|为数据集的总行数，|D(j)|为数据集按某个属性分类后各分类的行数，所有的对数都是以2为底的

1. 信息增益：使用香农提出的信息熵来度量一个数据集的纯度，这个算法是最古老的决策树算法称之为 ID3
             数据集的信息熵：Info(D) = -(p1*log(p1) + p2*log(p2) +...+pn*log(pn))
             pi是D中任意元素属于类Ci的概率，用|C(i, D)| / |D|来估计,
             根据属性A划分后的信息熵：
                               |D(j)|
             Info(A, D) = Sum(-------- * Info(D(j)) ), j就是A中不同元素的数量
                                |D|

             那么划分后的信息增益 Gain(A) = Info(D) - Info(A, D), 信息熵越小，纯度越高。
             我们尝试着用每个属性去划分数据集D，计算每个Gain(A(i))，然后选取最大的那个属性来划分D
             
2. 信息增益率：ID3算法偏向于选择分类数量最多的属性，假设采用数据集的行编号来划分数据集，因为每个编号都不一样，
             一次划分后就达到了最优解，但根据行编号来划分显然是没意义的。所以后人提出了另一种方法来改善这个问题
             它用分裂信息(split information)来将信息增益规范化，类似于Info(D):
                                   |D(j)|        |D(j)|
             SplitInfo(A, D) = -Sum( ------- * log(-------))
                                    |D|           |D|

                                   Gain(A, D)
             GainRate(A, D) = ---------------------
                                 SplitInfo(A, D)
                                 
             GainRate就是信息增益率，该算法称之为C4.5

3. 基尼指数：Gini Index, 基尼指数度量数据集的不纯度，该算法称之为CART

             Gini(D) = 1 - Sum(p(i)**2), pi是D中任意元素属于类Ci的概率，用|C(i, D)| / |D|来估计, 与ID3一致

             与前面两个不同的是，当用属性A划分D时，这个划分规则变得非常不一样，基尼指数考虑每个属性的二元划分
             假如A：{1, 2, 3}, 那么A的二元划分就是形如 {1, 2} - {3}这种，如果A有v个可能的取值，
             那么A就有2**v个可能的子集，不考虑满集和空集，存在(2**v - 2) / 2 种不同的划分，计算每种可能的划分：
                             |D(1)|                |D(2)|
             Gini(A(i), D) = ------ * Gini(D(1)) + ------ * Gini(D(2))
                               |D|                  |D|

             Delta(Gini(A)) = Gini(D) - Gini(A(i), D),
             选取最大化这个值的二元划分就好

关于这三种方法的综合评定，ID3倾向于选择多值属性，尽管增益率调整了这种便宜，但是它倾向于产生不平衡的划分，其中
一个分区比其他分区小得多。基尼指数偏向于多值属性，并且当类的数量很大时会有计算上的困难，因为这个是指数级的复杂度，
并且它还倾向于导致相等大小的分区和纯度。anyway，尽管这些算法都是有偏的，但是这些度量在实际中都产生了相当好的结果
             
'''

def calcShannonEnt(dataset):
    '''
    计算每个数据集的香农熵，ID3算法。以数据集最后一列提取分类标签
    '''
    dataset = np.array(dataset)
    rows, columns = np.shape(dataset)
    labels = dataset[:, -1]
    # 高效的写法
    label, counts = np.unique(labels, return_counts=True)
    labelCount = dict(zip(label, counts))
    # 累赘的写法
    '''
    labelCount = {}
    for i in range(len(labels)):
        if labels[i] not in labelCount:
            labelCount[labels[i]] = 1
        else:
            labelCount[labels[i]] += 1
    ent = 0
    '''
    ent = 0.0
    for label in labelCount:
        prob = labelCount[label] / rows
        ent -= prob*math.log(prob, 2)
    return ent

def splitDataset(dataset, axis, value):
    '''
    分割数据集，axis和value代表根据dataset中的第axis列，与value相等的每一行提取出来，组成新数据集
    '''
    dataset = np.array(dataset)
    resultDS = []
    for i in range(len(dataset)):
        data = dataset[i, axis]
        if data == value:
            row = np.delete(dataset[i], axis)
            resultDS.append(row)
    return np.array(resultDS)

def chooseBestToSplit(dataset, criterion='ID3'):
    '''
    选择最优的一列来进行分裂，返回该列的索引
    '''
    dataset = np.array(dataset)
    rowCount, columnCount = dataset.shape;
    featureCount = {}                       # key为每一列的不同元素，count为每个元素对应的个数
    gain = []                               # 存储根据每个feature进行划分的信息增益
    splitInfo = []                          # 存储每个特征列的splitInfo,用于C4.5，计算信息增益率
    ent_total = calcShannonEnt(dataset)     # 原始的信息熵
    for i in range(columnCount - 1):        # 最后一列是分类信息，所以不参与比较
        featureCount.clear()
        column = dataset[:, i]
        # 填充featureCount，遍历该列中每个元素
        labels, counts = np.unique(column, return_counts=True)
        featureCount = dict(zip(labels, counts))
        '''
        for j in range(rowCount):
            data = column[j]
            if data not in featureCount:
                featureCount[data] = 1
            else:
                featureCount[data] += 1
        '''
        ent_feature = 0.0
        iv = 0.0
        for key in featureCount:
            split_ds = splitDataset(dataset, i, key)    # 分裂数据集
            ent_split = calcShannonEnt(split_ds)        # 计算每个子数据集的信息熵
            weight = featureCount[key] / rowCount       # 计算每个子数据集的权重
            ent_feature += weight * ent_split           # 计算该列加权后的信息熵
            iv -= weight * math.log(weight, 2)          # 计算splitInfo

        gain.append(ent_total - ent_feature)            # 计算分裂前后的信息增益
        splitInfo.append(iv)
    if criterion == 'ID3':
        print('Information gain: ', gain)
        return gain.index(max(gain))
    else:
        gainRatio = np.divide(gain, splitInfo)
        sortedIndices = np.argsort(gainRatio)
        print('gainRatio: ', gainRatio)
        return sortedIndices[-1]

# 获取labels中出现频率最高的值
def majorityCount(labels):
    labelCounts = dict([(labels.count(label), label) for label in labels])
    return labelCounts(max(labelCounts.keys()))

def buildTree(dataset, feature_names, method='ID3'):
    dataset = np.array(dataset)
    labels = dataset[:, -1]
    if len(np.unique(labels)) == 1:
        return labels[0]
    if len(dataset[0]) == 1:
        return majorityCount(labels)
    bestFeature = chooseBestToSplit(dataset, criterion=method)
    bestFeatureName = feature_names[bestFeature]
    tree = {bestFeatureName: {}}
    feature_names = np.delete(feature_names, bestFeature)
    print(dataset[:, bestFeature])
    uniqueFeatureVals = np.unique(dataset[:, bestFeature])
    print(uniqueFeatureVals)
    for value in uniqueFeatureVals:
        subFeatureNames = feature_names[:]
        subDS = splitDataset(dataset, bestFeature, value)
        tree[bestFeatureName][value] = buildTree(subDS, subFeatureNames, method)

    return tree

# 使用pickle来持久化训练出来的决策树模型
def storeTree(inputTree, fileName):
    with open(fileName, 'w') as fw:
        pickle.dump(inputTree, fw)

# 从pickle出来的文件中恢复决策树模型
def grabTree(fileName):
    with open(fileName) as fr:
        return pickle.load(fileName)

def loadDataSet(path):
    records = []
    with open(path, 'rb') as fp:
        content = fp.read()
        rowlist = content.splitlines()
        records = [row.split('\t') for row in rowlist if row.strip()]
    return records

watermelon = [['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
              ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
              ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
              ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
              ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
              ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '是'],
              ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '是'],
              ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '是'],
              ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '否'],
              ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '否'],
              ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '否'],
              ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '否'],
              ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '否'],
              ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '否'],
              ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '否'],
              ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '否'],
              ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '否']]
print('watermelon dataset: \n', np.array(watermelon))
ent_all = calcShannonEnt(watermelon)
print('Entropy of watermelon: ', ent_all)

# print(splitDataset(watermelon, 5, '软粘'))
# print(chooseBestToSplit(watermelon))
feature_names = np.array(['色泽','根蒂','敲声','纹理','脐部','触感','好瓜'])
print('build Tree using ID3: \n', buildTree(watermelon, feature_names))
print('build Tree using C4.5: \n', buildTree(watermelon, feature_names, method='C4.5'))

