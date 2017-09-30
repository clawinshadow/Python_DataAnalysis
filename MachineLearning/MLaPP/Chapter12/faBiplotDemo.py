import numpy as np
import scipy.io as sio
import sklearn.decomposition as sd
import sklearn.preprocessing as spp
import matplotlib.pyplot as plt

def biPlot(C, Z, labels):
    '''
    biplot的画法有点奇葩，需要做很多格式化的预处理工作，用于在图形中获得更好的可读性，参见matlab的doc:
    https://www.mathworks.com/help/stats/biplot.html
    
    1. flip the sign: 找出每一列中的绝对值最大的那个元素，如果它是负数，则将这一列都乘以 -1 ，
                      迫使一列中绝对值最大的都为正数，而将一个向量乘以一个标量是不会影响它的方向的
    2. scale Z: 找出Z中绝对值最大的一个元素，和C中长度最长的一个值
                2.1: max(abs(Z))
                2.2: 计算每一行的L2范数，取最大的那一个
                然后将 Z * 2.2 / 2.1 来进行缩放
    '''
    # flip the sign
    D, L = C.shape
    N = Z.shape[0]
    W = np.zeros(C.shape)
    flipCols = []  # 记录哪些列的符号改变过，Z里面一样要用到
    for i in range(L):
        maxIndex = np.argmax(np.absolute(C[:, i]))
        maxAbs = C[maxIndex, i]
        if maxAbs < 0:
            flipCols.append(i)
            W[:, i] = C[:, i] * -1
        else:
            W[:, i] = C[:, i]

    # scale Z
    maxAbs = np.max(np.absolute(Z))
    maxLen = np.max(np.sum(C**2, axis=1)) ** 0.5
    print(maxAbs, maxLen)
    S = Z * maxLen / maxAbs
    for i in range(len(flipCols)):
        col = flipCols[i]
        S[:, col] = S[:, col] * -1  # C里面哪些翻转了符号的，Z里面要一样的处理

    # plot
    fig = plt.figure(figsize=(7, 6))
    fig.canvas.set_window_title('faBiplotDemo')

    ax = plt.subplot()
    plt.title('rotation = none')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    ax.grid(ls='dotted')
    ax.spines['top'].set_visible(False)   # Hide Border
    ax.spines['right'].set_visible(False)
    plt.axis([-1.1, 1.1, -1.1, 1.1])
    plt.xticks(np.linspace(-1, 1, 5))
    plt.yticks(np.linspace(-1, 1, 11))
    plt.plot([-1.1, 1.1], [0, 0], 'k-', lw=0.5)
    plt.plot([0, 0], [-1.1, 1.1], 'k-', lw=0.5)
    for i in range(len(W)):
        wi = W[i]
        label = labels[i]
        plt.plot(wi[0], wi[1], marker='o', color='darkblue', ms=2)
        plt.plot([0, wi[0]], [0, wi[1]], color='darkblue', lw=0.5)
        plt.annotate(label, xy=(wi[0], wi[1]), xytext=(wi[0] - 0.02, wi[1] + 0.02))

    plt.plot(S[:, 0], S[:, 1], 'ro', linestyle='none', ms=1)
    plt.show()

# load data
data = sio.loadmat('04cars.mat')
X = data['X'][:, 7:18]# use real-value features
y = data['names']
X = spp.StandardScaler().fit_transform(X)
labels = np.array(['Retail',
                   'Dealer',
                   'Engine',
                   'Cylinders',
                   'Horsepower',
                   'City MPG',
                   'Highway MPG',
                   'Weight',
                   'Wheel Base',
                   'Length',
                   'Width'])  # 每一列的标签
print('X.shape: ', X.shape)
print('y.shape: ', y.shape)

# fit FA model
L = 2
FA = sd.FactorAnalysis(n_components=L)
FA.fit(X)
C = FA.components_.T  # N * L
Z = FA.transform(X)
print('FA.W: \n', FA.components_)
print('psi: \n', FA.noise_variance_)
print('latent Z: \n', Z)

biPlot(C, Z, labels)
