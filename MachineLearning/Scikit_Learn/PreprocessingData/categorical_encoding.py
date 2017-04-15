import numpy as np
import sklearn.preprocessing as sp

'''
    对分类型特征列进行编码，例如一个人可能有如下特征['male', 'female'], ['from Europe', 'from US', 'from Asia']
, ["uses Firefox", "uses Chrome", "uses Safari", "uses Internet Explorer"].
    如果想要对这些特征进行编码，最有效的方式就是按1,2,3..进行编码，
    比如["male", "from US", "uses Internet Explorer"] -> [0, 1, 3]。
    然而这样有个缺点就是这种分类型的变量是无序的，他不是rating型变量，比如[A, B, C]等是有序的，
    如果按递增的数字0, 1, 2, 3去进行编码的话，就是将无序的分类型变量转化成有序的rating型变量了，样本将会失真
    sklearn中提供了一种编码方式叫OneHotEncoder, 假如分类变量有m种可能情况，则它的编码长度也是m，只包含0和1
    比如[male, female]中，male是[0, 1], female是[1, 0]
    ['from Europe', 'from US', 'from Asia'] -> [1, 0, 0, 0, 1, 0, 0, 0, 1]
    对每个分类变量来说，只有一位是1，其余全为0.
    这样的话占用了更多的存储空间，类似于稀疏矩阵，但是保证了分类变量的无序性
'''
enc = sp.OneHotEncoder()
# 第一列有2种情况，第二列3种情况，第三列4种情况
X = [[0, 0, 3],
     [1, 1, 0],
     [0, 2, 1],
     [1, 0, 2]]
enc.fit(X)
print('Categorical Feature X: \n', X)
X_test = np.array([[0, 1, 3]])
print('X_test: ')
print('Encoding X_test: ', enc.transform(X_test).toarray())

