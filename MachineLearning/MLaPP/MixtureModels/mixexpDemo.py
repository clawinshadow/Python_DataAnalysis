import numpy as np
import scipy.stats as ss
import scipy.linalg as sl
import matplotlib.pyplot as plt
import sklearn.linear_model as slm
import sklearn.preprocessing as sp

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
    for i in range(K):
        weight = V[i]
        p = np.exp(np.sum(weight.ravel() * x.ravel()))
        eta.append(p)
        total += p
        
    return (np.array(eta) / total).ravel()

def jac_hessian(X_train, y_train, V):
    D, K = X_train.shape[1], y_train.shape[1]
    g = np.zeros((K*D , 1))
    h = np.zeros((K*D, K*D))
    for i in range(len(X_train)):
        xi = X_train[i].reshape(-1, 1)
        yi = y_train[i].reshape(-1, 1)
        mui = Softmax(xi, V).reshape(-1, 1)
        gi = np.kron(mui - yi, xi)   # KD * 1
        g += gi
        hi = np.kron(np.diag(mui.ravel()) - np.dot(mui, mui.T), np.dot(xi, xi.T))
        h += hi

    return np.array(g).reshape(g.shape), np.array(h).reshape(h.shape)

# 自己写一个计算multiclass logistic regression的简单模型(没有用BFGS)
def MLG(X_train, y_train, maxIter=100):
    D = X_train.shape[1]
    K = y_train.shape[1]        # 假定y_train采用的是1-of-K coding
    V = 0 * np.ones((K*D, 1)) # 初始化权重矩阵V
    for i in range(maxIter):
        g, h = jac_hessian(X_train, y_train, V)
        print('g: ', g)
        print('h: \n', h)
        V = V - np.dot(sl.inv(h), g)
        print('V: \n', V.reshape(K, D))

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
def MStep(X_train, Y_train, r, weightMat, sigmaMat, mixCoef, K):
    weights = []
    sigmas = []
    y = Y_train.reshape(-1, 1)
    for i in range(K):
        Rk = np.diag(r[:, i].ravel())
        val1 = np.dot(X_train.T, Rk)
        weight_K = np.dot(sl.inv(np.dot(val1, X_train)), np.dot(val1, y))
        sigma_K = UpdateSigma(weight_K, X_train, y, Rk)

        weights.append(weight_K)
        sigmas.append(sigma_K)

    # use multinomial logistic regression to update V mat
    V = MLG(X_train, r)
    print('LG.coef_: ', LG.coef_)
    print('LG.intercept: ', LG.intercept_)

    return np.array(weights), np.array(sigmas), np.array(mixCoef)
        

def UpdateSigma(weight, X, y, r):
    total = 0
    for i in range(len(r)):
        xi, yi = X[i], y[i]
        ri = r[i]
        total += ri * (yi - np.sum(weight.ravel() * xi.ravel())) ** 2

    return (total / np.sum(r)) ** 0.5

def FitMixExp(X_train, Y_train, K, maxIter=50):
    # initial parameters， V和W是不一样的权重矩阵，虽然shape一样
    mixCoef = 0.1 * np.random.randn(K, X_train.shape[1])    # 书中的V矩阵， (K * D), 每一行都是一个D维的权重矩阵
    weightMat = 0.1 * np.random.randn(K, X_train.shape[1])  # 每个exp中线性回归的权重矩阵 (K * D)
    sigmaMat = np.random.rand(K)                            # 每个exp的标准差，注意是标准差 (K, )

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
# LG = slm.LogisticRegression(C=1e10, fit_intercept=False, multi_class='multinomial')
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
