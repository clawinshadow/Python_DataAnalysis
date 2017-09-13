import numpy as np
import scipy.stats as ss
import scipy.linalg as sl
import scipy.optimize as so
import matplotlib.pyplot as plt
import sklearn.linear_model as slm
import sklearn.preprocessing as sp

'''
需要牢记的是在MoE模型中，给定K个专家，那么 W 和 V 参数矩阵虽然shape都是K*D，但是其功能和意义都完全不一样
V 矩阵是给softmax函数使用的，W 矩阵相当于是线性回归的参数，
V 矩阵用于计算每个数据点属于每个K分类的概率， W 矩阵用于计算数据点在每个分类中的回归模型中的预测概率
最后就是一维数组的参数，每个K分类的方差
'''

# 计算每一行的rik， 为了防止下溢，使用log来计算
def GetResponsiblity(x, y, weightMat, sigmaMat, mixCoef, K):
    r = []
    total = 0
    pi = Softmax(x, mixCoef)
    for j in range(K):
        weight = weightMat[j]
        sigma = sigmaMat[j]
        mu = np.sum(weight.ravel() * x.ravel())
        prob = ss.norm(mu, sigma).pdf(y)

        val = pi[j] * prob
        r.append(val)
        total += val

    return (np.array(r) / total).ravel()

def Softmax(x, V):
    eta = []
    total = 0
    for i in range(len(V)):
        weight = V[i]
        p = np.exp(np.sum(weight.ravel() * x.ravel()))
        eta.append(p)
        total += p
        
    return (np.array(eta) / total).ravel()

def jac_hessian(X_train, y_train, V):
    D, K = X_train.shape[1], y_train.shape[1]
    g = np.zeros((K*D , 1))
    h = np.zeros((K*D, K*D))
    Vx = V.reshape(K, D)
    for i in range(len(X_train)):
        xi = X_train[i].reshape(-1, 1)
        yi = y_train[i].reshape(-1, 1)
        mui = Softmax(xi, Vx).reshape(-1, 1)
        gi = np.kron(mui - yi, xi)   # KD * 1
        g += gi
        hi = np.kron(np.diag(mui.ravel()) - np.dot(mui, mui.T), np.dot(xi, xi.T))
        h += hi

    return np.array(g).reshape(g.shape), np.array(h).reshape(h.shape)

# 自己写一个计算multiclass logistic regression的简单模型(没有用BFGS)
def MLG(X_train, y_train, maxIter=100, C=0.001):
    D = X_train.shape[1]
    K = y_train.shape[1]        # 假定y_train采用的是1-of-K coding
    V = 0.1 * np.random.rand(K*D, 1) # 初始化权重矩阵V
    for i in range(maxIter):
        g, h = jac_hessian(X_train, y_train, V)
        # L2 penalize，logistic回归的话，如果数据是线性可分的，并且不加penalize的话，会有无穷多个解
        # 并且随着迭代次数的增多，w会越来越大，最终overflow
        g = g + C * V      
        h = h + C * np.eye(K*D)
        V = V - np.dot(sl.inv(h), g)

    return V.reshape(K, D)

# 计算rik
def EStep(X_train, Y_train, weightMat, sigmaMat, mixCoef, K):
    r = []
    for i in range(len(X_train)):
        xi = X_train[i]
        yi = Y_train[i]
        r.append(GetResponsiblity(xi, yi, weightMat, sigmaMat, mixCoef, K))

    return np.array(r)  # N * K

# 更新参数
def MStep(X_train, Y_train, r):
    weights = []
    sigmas = []
    y = Y_train.reshape(-1, 1)
    for i in range(r.shape[1]):
        Rk = np.diag(r[:, i].ravel())
        val1 = np.dot(X_train.T, Rk)
        weight_K = np.dot(sl.inv(np.dot(val1, X_train)), np.dot(val1, y))
        sigma_K = UpdateSigma(weight_K, X_train, y, Rk)

        weights.append(weight_K.ravel())
        sigmas.append(np.asscalar(sigma_K))

    # use multinomial logistic regression to update V mat
    V = MLG(X_train, r)
    print('V: \n', V)

    return np.array(weights), np.array(sigmas), np.array(V)
        
def UpdateSigma(weight, X, y, r):
    total = 0
    for i in range(len(r)):
        xi, yi = X[i], y[i]
        ri = r[i, i]
        total += ri * (yi - np.sum(weight.ravel() * xi.ravel())) ** 2

    return (total / np.sum(r)) ** 0.5

def FitMixExp(X_train, Y_train, K, maxIter=40):
    # initial parameters， V和W是不一样的权重矩阵，虽然shape一样
    mixCoef = 0.1 * np.random.randn(K, X_train.shape[1])    # 书中的V矩阵， (K * D), 每一行都是一个D维的权重矩阵
    weightMat = 0.1 * np.random.randn(K, X_train.shape[1])  # 每个exp中线性回归的权重矩阵 (K * D)
    sigmaMat = np.random.rand(K)                            # 每个exp的标准差，注意是标准差 (K, )

    for i in range(maxIter):
        print('{0:-^60}'.format('Iteration: ' + str(i + 1)))
        r = EStep(X_train, Y_train, weightMat, sigmaMat, mixCoef, K)
        weightMat, sigmaMat, mixCoef = MStep(X_train, Y_train, r)
        print('W: \n', weightMat)
        print('sigmaMat: \n', sigmaMat.ravel())
        print('V: ', mixCoef)

    return mixCoef, weightMat, sigmaMat     

np.random.seed(0)
# generate data
w = 0.01 * np.random.randn(3)
b = [-1, 1, -1]
x = np.linspace(-1, 1, 50)
y = []
for i in range(len(x)):
    xi = x[i]
    yi = 0
    if xi <= -0.5:
        yi = w[0] * xi + b[0]
    elif xi <= 0.5:
        yi = w[1] * xi + b[1]
    else:
        yi = w[2] * xi + b[2]

    y.append(yi)

y = np.array(y) + 0.2 * np.random.randn(len(x))

# Fit Model
# LG = slm.LogisticRegression(C=1e10, fit_intercept=False, multi_class='multinomial')
x_train = x.reshape(-1, 1)
SS = sp.StandardScaler()
SS.fit(x_train)    # 记住参数
x_train = SS.transform(x_train)
x_train = np.c_[np.ones(len(x_train)).reshape(-1, 1), x_train]
y_train = y.reshape(-1, 1)

K = 3
mixCoef, weightMat, sigmaMat = FitMixExp(x_train, y_train, K)

# plot training points
fig = plt.figure(figsize=(11, 10))
fig.canvas.set_window_title('mixexpDemo')

plt.subplot(221)
plt.title('expert predictions, fixed mixing weights: False')
plt.axis([-1, 1, -1.5, 1.5])
plt.xticks(np.linspace(-1, 1, 5))
plt.yticks(np.linspace(-1.5, 1.5, 7))
plt.plot(x, y, 'bo', ls='none', fillstyle='none')

# plot the regression line
xx = np.linspace(-1, 1, 200)
xx_standard = SS.transform(xx.reshape(-1, 1))
xx_standard = np.c_[np.ones(len(xx_standard)).reshape(-1, 1), xx_standard]
yy = np.dot(xx_standard, weightMat.T)
plt.plot(xx, yy[:, 0], 'r:')
plt.plot(xx, yy[:, 1], 'b-')
plt.plot(xx, yy[:, 2], 'k-.')

# plot the gating functions
yy2 = []
for i in range(len(xx_standard)):
    xi = xx_standard[i]
    yy2.append(Softmax(xi, mixCoef))
yy2 = np.array(yy2).reshape(yy.shape)

plt.subplot(222)
plt.title('gating functions, fixed mixing weights: False')
plt.axis([-1.1, 1.1, -0.2, 1.2])
plt.xticks(np.linspace(-1, 1, 5))
plt.yticks(np.linspace(0, 1, 11))
plt.plot(xx, yy2[:, 0], 'r:')
plt.plot(xx, yy2[:, 1], 'b-')
plt.plot(xx, yy2[:, 2], 'k-.')

# plot the predicts
muk = np.dot(x_train, weightMat.T)
mus = []
sigmas = []
sigmaMat = sigmaMat.ravel()
for i in range(len(x_train)):
    pi = Softmax(x_train[i], mixCoef).ravel()
    muki = muk[i].ravel()
    mui = np.sum(mui * pi)
    mus.append(mui)  # 均值
    sigmas.append(np.sum(pi * (sigmaMat + muki**2)) - mui**2)

sigmas = np.array(sigmas) ** 0.5

plt.subplot(223)
plt.title('predicted mean and var, fixed mixing weights: False')
plt.axis([-1.1, 1, -1.6, 1.5])
plt.xticks(np.linspace(-1, 1, 5))
plt.yticks(np.linspace(-1.5, 1.5, 7))
plt.plot(x, y, 'bo', ls='none', fillstyle='none')
plt.plot(x, mus, 'r-')
plt.errorbar(x, mus, mus+sigmas, mus-sigmas, ecolor='midnightblue')

plt.show()
