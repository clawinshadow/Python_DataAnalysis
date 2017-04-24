import numpy as np
import sklearn.model_selection as sm

'''
demos about split traning & testing dataset from sample data.
'''

# train_test_split 用于在样本数据中随机抽取，一般test_size和train_size只提供一个就够了
# 如果是0-1之间的浮点数，则解释为比例，如果是整数，则理解为绝对数目，stratify参数表明是否按分层抽样
X = np.arange(40).reshape((20, 2))                  # 样本数据
y = np.hstack((np.ones((10, )), np.zeros((10,))))   # 样本标签
print('X: \n', X)
print('y: ', y)
X_train, X_test, y_train, y_test = sm.train_test_split(X, y, test_size=0.3) # testset占比0.3， 6个
print('X_train: \n', X_train)
print('X_test: \n', X_test)
print('y_train: \n', y_train)
print('y_test: \n', y_test)

X_train, X_test, y_train, y_test = sm.train_test_split(X, y, test_size=5) # testset的绝对数量为5个
print('X_train: \n', X_train)
print('X_test: \n', X_test)
print('y_train: \n', y_train)
print('y_test: \n', y_test)

# 按y中的标签的占比进行分层抽样，1和0各占一半，那么结果中应该各含三个
X_train, X_test, y_train, y_test = sm.train_test_split(X, y, test_size=6, stratify=y)
print('X_train: \n', X_train)
print('X_test: \n', X_test)
print('y_train: \n', y_train)
print('y_test: \n', y_test)

print('{0:-^70}'.format('K-Fold'))
# K-Fold, 著名的K折法分割数据集
X = np.array([[1, 2], [3, 4], [1, 4], [2, 3], [2, 5], [3, 7], [6, 1], [3, 8]])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8])
kf =sm.KFold(n_splits=4) # 默认的shuffle是False，不打乱数据，按顺序分割数据
print('X: \n', X)
print('y: ', y)
print('KFold class: ', kf)
print('splits of kf: ', kf.get_n_splits(X))
for train_indices, test_indices in kf.split(X):
    print('Train Indices: ', train_indices, 'Test Indices: ', test_indices)

kf = sm.KFold(n_splits=4, shuffle=True) # 打乱数据，不再是[0, 1], [2, 3], [4, 5], [6, 7]了
print('Shuffle KFold class:', kf)
print('splits of kf: ', kf.get_n_splits(X))
for train_indices, test_indices in kf.split(X):
    print('Train Indices: ', train_indices, 'Test Indices: ', test_indices)

# 当样本总量N_samples不能整除n_splits时，按如下逻辑分割数据集
# The first N_samples % n_splits folds have size: n_samples // n_splits + 1,
# other folds have size: n_samples // n_splits
kf = sm.KFold(n_splits=3)      # should be [3, 3, 2]
print('KFold class:', kf)
print('splits of kf: ', kf.get_n_splits(X))
for train_indices, test_indices in kf.split(X):
    print('Train Indices: ', train_indices, 'Test Indices: ', test_indices)

print('{0:-^70}'.format('Leave One Out'))
# LOO, 只保留一个数据作为测试数据，对于N个样本量的数据来说，它就有N种划分
# 对于大数据来说，这个是非常耗时的，要注意不要滥用
loo = sm.LeaveOneOut()
print('Leave One Out class: ', loo)
print('splits of loo: ', loo.get_n_splits(X))
for train_indices, test_indices in loo.split(X):
    print('Train Indices: ', train_indices, 'Test Indices: ', test_indices)

print('{0:-^70}'.format('Leave P Out'))
# LPO，LOO的一般形式，LOO是只留出一个作为测试数据，LPO是留出P个作为测试数据
# 与KFold不同的是，LPO留出所有可能的P个组合，总共有C(N, p)种情况，复杂度也是非常高
lpo = sm.LeavePOut(2)
print('Leave P Out class: ', lpo)
print('splits of lpo: ', lpo.get_n_splits(X))  # Combine(8, 2) = 7*8/2 = 28
for train_indices, test_indices in lpo.split(X):
    print('Train Indices: ', train_indices, 'Test Indices: ', test_indices)

print('{0:-^70}'.format('Shuffle Split'))
# 就是打乱数据后随机分割数据集，与第一个方法类似
ss = sm.ShuffleSplit(n_splits=3, test_size=0.25, random_state=0)
print('Shuffle Split class: ', ss)
print('splits of ss: ', ss.get_n_splits(X))  # 就是构造函数中的n_splits参数
for train_indices, test_indices in ss.split(X):
    print('Train Indices: ', train_indices, 'Test Indices: ', test_indices)

# 以上的K_Fold, LOO, LPO, Shuffle等方法都是建立在样本数据独立同分布的基础上的(i.d.d)
# 只有在i.d.d的基础上，以上抽样方法才能保留数据的统计特征，做到不失真
# 以下是数据分布不均匀的时候，基于分层或分组的抽样方法
print('{0:-^70}'.format('Stratified K-Fold'))
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
skf = sm.StratifiedKFold(n_splits=4)
print('Stratified K-Fold class: ', skf) 
print('splits of skf: ', skf.get_n_splits(X, y))  # 增加一个参数，根据y来分层
for train_indices, test_indices in skf.split(X, y):
    print('Train Indices: ', train_indices, 'Test Indices: ', test_indices)

# Group K-Fold, 除了X和y之外，还有个额外的参数是每个样本所属的组
# 抽样要保证测试集里面的数据所属的组与训练集里面的样本所属的组是完全不一样的
print('{0:-^70}'.format('Group K-Fold'))
X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 8.8, 9, 10]
y = ["a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]
groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
gkf = sm.GroupKFold(n_splits=3)  # groups的数量必须要大于n_splits
print('X: \n', X)
print('y: ', y)
print('groups: ', groups)
print('Group K-Fold class: ', gkf) 
print('splits of gkf: ', gkf.get_n_splits(X, y, groups))  # 再增加一个分组的参数
for train_indices, test_indices in gkf.split(X, y, groups):
    print('Train Indices: ', train_indices, 'Test Indices: ', test_indices)

# Leave One Group out
print('{0:-^70}'.format('Leave One Group out'))
logo = sm.LeaveOneGroupOut()  
print('Leave One Group out class: ', logo) 
print('splits of logo: ', logo.get_n_splits(X, y, groups=groups))  # 等于groups的数量
for train_indices, test_indices in logo.split(X, y, groups=groups):
    print('Train Indices: ', train_indices, 'Test Indices: ', test_indices)

# Leave P Groups out
print('{0:-^70}'.format('Leave P Groups out'))
groups = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]   # 共5组
lpgo = sm.LeavePGroupsOut(n_groups=2)     
print('Leave P Groups out class: ', lpgo) 
print('splits of lpgo: ', lpgo.get_n_splits(X, y, groups=groups))  # Combine(5, 2)  = 10
for train_indices, test_indices in lpgo.split(X, y, groups=groups):
    print('Train Indices: ', train_indices, 'Test Indices: ', test_indices)

# Group Shuffle Split
print('{0:-^70}'.format('Group Shuffle Split'))
gss = sm.GroupShuffleSplit(n_splits=4, test_size=0.5, random_state=0)     
print('Group Shuffle Split class: ', gss) 
print('splits of gss: ', gss.get_n_splits(X, y, groups=groups))  # 等于n_splits
for train_indices, test_indices in gss.split(X, y, groups=groups):
    print('Train Indices: ', train_indices, 'Test Indices: ', test_indices)

# Time Series Split, 按时间顺序的分割数据，训练数据集必须是连续的，不能打乱，只留出最后的几个为测试数据
print('{0:-^70}'.format('Time Series Split'))
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4, 5, 6])
print('Time Series X: \n', X)
print('y: ', y)
tscv = sm.TimeSeriesSplit(n_splits=3) # 留出最后3个作为
print('Time Series Split class: ', tscv)  
for train_indices, test_indices in tscv.split(X):
    print('Train Indices: ', train_indices, 'Test Indices: ', test_indices)

