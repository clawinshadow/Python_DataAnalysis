import numpy as np
import scipy.stats as ss
import scipy.linalg as sl
import matplotlib.pyplot as plt
import sklearn.linear_model as slm
import sklearn.preprocessing as sp

# 计算每一行的rik
def GetResponsiblity(x, y, weightMat, sigmaMat, mixCoef, K):
    r = []
    total = 0
    for i in range(K):
        weight = weightMat[i]
        sigma = sigmaMat[i]
        mix = mixCoef[i]
        mu = np.sum(weight.ravel() * x.ravel())
        prob = ss.norm(mu, sigma).pdf(y)

        val = mix * prob
        r.append(val)
        total += val

    return (r / total).ravel()

# 计算rik
def EStep(X_train, Y_train, weightMat, sigmaMat, mixCoef, K):
    r = []
    for i in range(len(X_train)):
        xi = X_train[i]
        yi = Y_train[i]
        r.append(GetResponsiblity(xi, yi, weightMat, sigmaMat, mixCoef, K))

    return np.array(r)  # N * K

# 更新参数
def MStep(X_train, Y_train, r, weightMat, sigmaMat, mixCoef, K):
    weights = []
    sigmas = []
    mixCoef = np.mean(r, axis=1)
    y = Y_train.reshape(-1, 1)
    for i in range(K):
        Rk = np.diag(r[:, i].ravel())
        val1 = np.dot(X_train.T, Rk)
        print(val1)
        print(np.dot(val1, X_train))
        print(sl.inv(np.dot(val1, X_train)))
        weight_K = np.dot(sl.inv(np.dot(val1, X_train)), np.dot(val1, y))
        sigma_K = UpdateSigma(weight_K, X_train, y, Rk)

        weights.append(weight_K)
        sigmas.append(sigma_K)

    return np.array(weights), np.array(sigmas), np.array(mixCoef)
        

def UpdateSigma(weight, X, y, r):
    total = 0
    for i in range(len(r)):
        xi, yi = X[i], y[i]
        ri = r[i]
        total += ri * (yi - np.sum(weight.ravel() * xi.ravel())) ** 2

    return (total / np.sum(r)) ** 0.5

def FitMixExp(X_train, Y_train, K, maxIter=50):
    # initial parameters
    mixCoef = np.tile(1/K, K)                              # 混合系数 pi， (K, )
    weightMat = 0.1 * np.random.randn(K, X_train.shape[1]) # 每个exp中线性回归的权重矩阵 (K * D)
    sigmaMat = np.random.rand(K)                           # 每个exp的标准差，注意是标准差  (K, )

    for i in range(maxIter):
        print('Iteration: ', str(i + 1))
        r = EStep(X_train, Y_train, weightMat, sigmaMat, mixCoef, K)
        print('r: \n', r)
        weightMat, sigmaMat, mixCoef = MStep(X_train, Y_train, r, weightMat, sigmaMat, mixCoef, K)
        print('weights: \n', weightMat)
        print('sigmaMat: \n', sigmaMat)
        print('mixCoef: ', mixCoef)
        print('{0:-^60}'.format(''))

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
# LG = slm.LogisticRegression(C=1e10, fit_intercept=False, multiclass='multinomial')
x = x.reshape(-1, 1)
SS = sp.StandardScaler()
SS.fit(x)    # 记住参数
x = SS.transform(x)
X_train = np.c_[np.ones(len(x)).reshape(-1, 1), x]
y_train = y.reshape(-1, 1)

K = 3
mixCoef, weightMat, sigmaMat = FitMixExp(X_train, y_train, K)

# plots
fig = plt.figure(figsize=(11, 10))
fig.canvas.set_window_title('mixexpDemo')

plt.subplot(221)
plt.axis([-1, 1, -1.5, 1.5])
plt.xticks(np.linspace(-1, 1, 5))
plt.yticks(np.linspace(-1.5, 1.5, 7))
plt.plot(x, y, 'bo', ls='none', fillstyle='none')

plt.show()
