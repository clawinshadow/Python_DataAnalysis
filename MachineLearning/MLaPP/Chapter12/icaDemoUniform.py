import numpy as np
import sklearn.decomposition as sd
import matplotlib.pyplot as plt

'''
PCA中的W是正交规范化的，它发现的是X在哪些方向上的投影拥有最大的方差，这些方向是彼此正交的，但是它的
不可辨别性限制了它无法辨识出原始的信号是什么，它最多也只能把数据漂白，即剔除feature之间的相关性。
ICA相当于进阶版的PCA，在概率模型里面它放宽了prior必须为正态分布的限制，并且混合矩阵W不再要求正交规范性，
转而要求在每个方向上的投影的方差均为1，相等的方差哦。这样一来unidentifiability就不存在了，通过FastICA
可以还原出原始信号。
'''

np.random.seed(2)

N = 100
A = 0.3 * np.array([[2, 3],
                    [2, 1]]) # mixing matrix

X = np.random.rand(N, 2) * 4 - 2  # Uniform ~ [-2, 2]
X_train = np.dot(X, A)

# restore with PCA
N, D = X_train.shape
pca = sd.PCA(whiten=True).fit(X_train)
X_PCA = pca.transform(X_train)
print(X_PCA.shape)

# restore with ICA
ica = sd.FastICA(whiten=False)
X_ICA = ica.fit_transform(X_PCA)
print('mixing matrix by ICA: \n', ica.mixing_)

# plots
fig = plt.figure(figsize=(11, 9))
fig.canvas.set_window_title('icaDemoUniform')

def basisAttr():
    plt.axis([-4, 4, -3, 3])
    plt.xticks(np.linspace(-3, 3, 7))
    plt.yticks(np.linspace(-3, 3, 7))
    plt.plot([-3, 3], [0, 0], 'k-', lw=2)
    plt.axvline(0, color='k', lw=2)
    
plt.subplot(221)
basisAttr()
plt.title('uniform data', fontdict={'fontsize': 10})
plt.plot(X[:, 0], X[:, 1], color='midnightblue', linestyle='none', marker='o', ms=4)

plt.subplot(222)
basisAttr()
plt.title('uniform data after linear mixing', fontdict={'fontsize': 10})
plt.plot(X_train[:, 0], X_train[:, 1], color='midnightblue', linestyle='none', marker='o', ms=4)

plt.subplot(223)
basisAttr()
plt.title('PCA applied to mixed data from uniform source', fontdict={'fontsize': 10})
plt.plot(X_PCA[:, 0], X_PCA[:, 1], color='midnightblue', linestyle='none', marker='o', ms=4)

plt.subplot(224)
basisAttr()
plt.title('ICA applied to mixed data from uniform source', fontdict={'fontsize': 10})
plt.plot(X_ICA[:, 0], X_ICA[:, 1], color='midnightblue', linestyle='none', marker='o', ms=4)

plt.show()
