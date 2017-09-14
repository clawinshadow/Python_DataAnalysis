import numpy as np
import scipy.stats as ss
import scipy.linalg as sl

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

    return np.array(weights), np.array(sigmas), np.array(V)
        
def UpdateSigma(weight, X, y, r):
    total = 0
    for i in range(len(r)):
        xi, yi = X[i], y[i]
        ri = r[i, i]
        total += ri * (yi - np.sum(weight.ravel() * xi.ravel())) ** 2

    return (total / np.sum(r)) ** 0.5

def FitMixExp(X_train, Y_train, K, maxIter=20):
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
