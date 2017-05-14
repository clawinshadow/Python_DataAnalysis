from math import log
import numpy as np
import scipy.stats as ss
import sklearn.feature_selection as sfs

'''
所谓特征选择，是在所有特征列里面按特定标准选取符合要求的特征列，它不是降维，只是选择合适的，剔除剩余的特征列

关于互信息 Mutual Information, 这是一个不太好理解的概念，用于度量两个随机变量X和Y之间共有的信息，假如X的信息熵
为H(X), 在观测到Y后X的条件熵为H(X|Y), 则互信息 I(X, Y) = H(X) - H(X|Y) = H(Y) - H(Y|X). 简单点来说计算方式如下：

对于离散性随机变量 X 与 Y:
                              P(X, Y)
I(X, Y) = Σ Σ P(X, Y) * log(-----------)
                             P(X)*P(Y)
对于连续型的随机变量，将求和符号改为积分即可
sklearn里面用自然对数作为对数的底，有时候用2为对数，10为对数都行。 I(X, Y)一定是大于等于零的，并且当且只当X与Y
相互独立时才等于零，这个值越大，表明X与Y的共享信息越多. sklearn似乎还支持对于连续型随机变量计算MI，但具体算法
还不清楚，留待以后查明。

'''                           

# VarianceThreshold 最简单的一个
print('{0:-^70}'.format('VarianceThreshold'))
x = np.array([[1, 0, 1],
              [2, 1, 0],
              [3, 1, 7],
              [4, 0, 4],
              [5, 0, 3],
              [6, 1, 6]])
print('X: \n', x)
print('x.shape: ', x.shape)
rows, columns = x.shape
for i in range(columns):
    feature = x[:, i]
    print('variance of {0} feature in x: {1}'.format(i, np.var(feature)))

# 根据方差来筛选特征，所有方差小于threshold的都将被剔除出去
sel = sfs.VarianceThreshold(threshold=1.0)
x_selected = sel.fit_transform(x)
print('VarianceThreshold class: ', sel)
print('feature selection from x: \n', x_selected)

# 单因素方差分析 One-way analysis of variance, 具体可参见wikipedia
print('{0:-^70}'.format('f_classif'))
data = np.array([1, 2, 3, 4, 5, 6])
sample = data[:, np.newaxis]                     # 构建训练数据，sample只包含一个特征列
label = np.array([0, 0, 0, 1, 1, 1])             # sample对应的分类标签
sst = np.var(data) * len(data)                   # 所有数据的总平方和
mean = np.mean(data)                             # 总体数据的均值

data_1 = np.extract(label == 0, data)            # 抽取所有分类为0的数据，这就是一组
data_2 = np.extract(label == 1, data)            # 抽取所有分类为1的数据
mean_1 = np.mean(data_1)
mean_2 = np.mean(data_2)                         # 计算两个分组各自的均值
sse = np.var(data_1) * 3 + np.var(data_2) * 3    # 计算组内平方和

# 计算组间平方和的时候一定要注意，每个组的平方和之前还要乘以每组内元素的个数
ssa = 3 * (mean_1 - mean)**2 + 3 * (mean_2 - mean)**2  
print('data: ', data)
print('label corrsponding to data: ', label)
print('SST, SSE, SSA: ', sst, sse, ssa)
print('SST = SSE + SSA : ', np.allclose(sst, sse + ssa))

# 计算自由度，假设数据总数为n，分组数为k，则组间平方和的自由度k-1， 组内平方和的为n-k，总体自由度为n-1
f_ssa = 2 - 1
f_sse = 6 - 2
msa = ssa / f_ssa     # msa就是组间均方，或者称为组间方差
mse = sse / f_sse     # mse就是组内方差
F = msa / mse         # F值是msa与mse的比值
print('F value: ', F)

# F服从自由度为(f_ssa, f_sse)的F分布，可用于进行假设检验
rv = ss.f(f_ssa, f_sse)
P = 1 - rv.cdf(F)
print('P Value: ', P)

# 使用f_classif来计算, 返回结果中就包含F和P值，非常方便
F, pval = sfs.f_classif(sample, label.T)
print('F, pval by sklearn: ', F, pval)

# chi2 卡方检验，卡方检验可用于拟合优度检验和列联表分析，但是chi2方法貌似只能用于拟合优度检验
print('{0:-^70}'.format('chi2'))
# 假设以下数据是一个六面骰子丢掷60次后每一面所得的结果，我们用拟合优度检验来判别该骰子是否是有偏的
data = np.array([5, 8, 9, 8, 10, 20])
sample = data[:, np.newaxis]
label = [1, 2, 3, 4, 5, 6]
fe = np.array([10, 10, 10, 10, 10, 10])   # 如果是无偏情况下的期望数据
chi2 = np.sum((data - fe)**2 / fe)
print('thoeritical chi2: ', chi2)
chi2, pval = sfs.chi2(sample, np.transpose(label))
print('chi2, pval by sklearn: ', chi2, pval)

# mutual information
# 假设投掷一个骰子，X是一个随机变量，X = 0表明骰子的点数为奇数， X = 1表明为偶数
# Y 也是一个随机变量，Y = 0表明骰子的点数为合数，Y = 1表明为质数
print('{0:-^70}'.format('Mutual Information'))
data = np.array([[0, 0],
                 [1, 1],
                 [0, 1],
                 [1, 0],
                 [0, 1],
                 [1, 0]])
# 列举出X和Y所有的取值范围，从中可计算出X和Y各自的概率分布以及联合概率分布
X = data[:, 0, np.newaxis]    
y = data[:, 1]                
print('Random Variable X: ', X)
print('Random Variable Y: ', y)
# X = 0 时骰子点数为1, 3, 5, =1 时为2, 4, 6, Y = 0时骰子为1, 4, 6, Y = 1时为2, 3, 5
# 所以P(X = 0) = P(X = 1) = P(Y = 0) = P(Y = 1) = 1/2
# 并且可以计算出联合概率分布：P(X = 0, Y = 0) = 1/6,
# P(X = 0, Y = 1) = 2/6, P(X = 1, Y = 0) = 2/6, P(X = 1, Y = 1) = 1/6,
# 所以I(X, Y) = P(X = 0, Y = 0) * ln(P(X = 0, Y = 0) / [P(X = 0) * P(Y = 0)]) +
#               P(X = 1, Y = 0) * ln(P(X = 1, Y = 0) / [P(X = 1) * P(Y = 0)]) +
#               P(X = 0, Y = 1) * ln(P(X = 0, Y = 1) / [P(X = 0) * P(Y = 1)]) +
#               P(X = 1, Y = 1) * ln(P(X = 1, Y = 1) / [P(X = 1) * P(Y = 1)])
I = 1/6 * log(4/6) + 2/6 * log(8/6) + 1/6 * log(4/6) + 2/6 * log(8/6)
print('Mutual Information by myself: ', I)

# 用 sklearn 来计算
MI = sfs.mutual_info_classif(X, y, discrete_features=True)
print('Mutual Information by sklearn: ', MI)
print('MI = I: ', np.allclose(I, MI))


