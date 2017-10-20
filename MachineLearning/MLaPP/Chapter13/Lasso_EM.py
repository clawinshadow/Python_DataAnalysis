import numpy as np
import scipy.io as sio
import scipy.stats as ss
import scipy.linalg as sl
import sklearn.linear_model as slm

'''
要注意两点
1. 对数据集进行预处理，Standardized X, Centered y
2. 要注意三个lambda的度量，主要是各自的目标函数不一样
   2.1: sklearn 里面的是 (1 / 2*N) * RSS(w) + λ1 * |w|_1
   2.2: coordinate descent 示例中使用的是 RSS(w) + λ * |w|_1
   2.3: 本例中EM 使用的是：NLL(w) + λ2*|w|_1 = (1 / 2*σ2) * RSS(w) + λ2 * |w|_1
   
   所以当我们要跟sklearn中的Lasso进行验证的时候，一定要rescale λ
   
   以2.2中的 λ 为基准，sklearn中的alpha = λ / 2N, EM中的是 λ / 2*σ2
   反之，如果我们以EM中的 λ2 为基准，则2.2中的 λ = 2 * σ2 * λ2, σ2是EM最后一步中对noise的计算结果
   那么送入sklearn中验证的参数即是 λ1 = λ / 2N = σ2 * λ2 / N
'''

# prepare data
data = sio.loadmat('prostateStnd.mat')
print(data.keys())
X, y = data['Xtrain'], data['ytrain']
y = y - np.mean(y)  # 下面EM的算法默认是Standardized X, Centered y

# Fit model with sklearn
N, D = X.shape

larsRes = slm.lars_path(X, y.ravel())
lambda_path = larsRes[0]      # 9个
coef_path = larsRes[2]        # (8, 9) every column is as w vector

# Fit model with EM
l = lambda_path[3] * 2 * N    # choose one lambda to be used in EM

def EStep(X, y, w, l):
    # calculate expectation of τ2 & σ2, 一个是GSM里面的参数，与w等长，是个数列，一个是noise
    e_tau2 = l / np.abs(w)

    N, D = X.shape
    y = y.reshape(-1, 1)
    w = w.reshape(-1, 1)
    a, b = 1, 1 # assume the prior of σ2 ~ IG(1, 1)
    residual = y - np.dot(X, w)
    an = a + N / 2
    bn = b + 0.5 * np.dot(residual.T, residual)
    e_delta2 = an / bn

    return e_tau2, e_delta2

def MStep(e_tau2, e_delta2, X, y):
    # re-estimate a new w
    d = X.shape[1]
    phi = sl.inv(np.diag(e_tau2.ravel()))
    U, D, Vt = sl.svd(X, full_matrices=False)  # D: (8, ), U: (67, 8)
    D1 = sl.diagsvd(D**-1, d, d)
    D2 = sl.diagsvd(D**-2, d, d)

    val1 = np.dot(phi, Vt.T)
    val2 = np.dot(Vt, np.dot(phi, Vt.T)) + (1 / e_delta2) * D2
    val3 = np.dot(D1, np.dot(U.T, y))

    w_new = np.dot(val1, np.dot(sl.inv(val2), val3))
    return w_new

def EM(X, y, l, maxIter=1000):
    N, D = X.shape
    w = np.tile(1/D, D)
    for i in range(maxIter):
        e1, e2 = EStep(X, y, w, l)
        w_new = MStep(e1, e2, X, y)
        print(w_new.ravel())
        if np.allclose(w, w_new):
            return w_new, e2

        w = w_new

    return w, e2  # w 和 σ2 的值

w, e2 = EM(X, y, l)
print('w by EM: ', w.ravel())
l2 = (lambda_path[3] * 2) / e2  # σ2 = 1 / e2
lasso = slm.Lasso(alpha=l2, fit_intercept=False).fit(X, y)
print('w by Lasso: ', lasso.coef_)
