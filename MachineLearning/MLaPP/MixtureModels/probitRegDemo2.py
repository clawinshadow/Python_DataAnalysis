import numpy as np
import scipy.stats as ss
import scipy.linalg as sl
import scipy.optimize as so
import matplotlib.pyplot as plt

'''
因为 1 - cdf(x) = cdf(-x), 所以有：
NLL(X, y, w) = Σ(ss.norm().cdf(yi * w.T*xi)) + lambda * np.sum(w**2), yi ~ {-1, +1}

用两种方法来Fit probit regression，普通的梯度方法和EM算法，都是MAP估计
1. Fit probit regression with gradient-based method
              yi * φ(w.T*xi)
   gi = xi * ---------------- , φ(x)是pdf，Φ(x)是cdf
               Φ(yi*w.T*xi)

                  φ(w.T*xi)**2       yi * w.T*xi * φ(w.T*xi)
   Hi = -xi * (----------------- + ---------------------------) * xi.T
                Φ(yi*w.T*xi)**2            Φ(yi*w.T*xi)

   regularizer: lambda = 0.1

   g = Σgi + 2 * lambda * w
   H = ΣHi + 2 * lambda * I

   w.next = w − sl.inv(H) * g


'''
np.random.seed(0)

# Generate data
N, D= 100, 2
X = np.random.randn(N, D)
X = np.c_[np.ones(N).reshape(-1, 1), X]
w = np.random.randn(X.shape[1])
y = []
for i in range(len(X)):
    val = np.sum(w * X[i])
    if val > 0:
        y.append(1)
    else:
        y.append(-1)
y = np.array(y)
lambdaVec = 0.1 * np.ones(X.shape[1])
lambdaVec[0] = 0  # don't penalize the biar term
lambdaVec = lambdaVec.reshape(-1, 1)
print('true w: ', w)

# Fit with gradient-based method
def NLL(X, y, w, Lambda):
    y1 = y.reshape(-1, 1)
    w1 = w.reshape(-1, 1)
    logli = -np.sum(ss.norm().logcdf(y1 * np.dot(X, w1)))
    return logli + np.sum(Lambda * w1**2)

def jac(X, y, w, Lambda):
    y1 = y.reshape(-1, 1)  # N * 1
    w1 = w.reshape(-1, 1)  # D * 1
    mu = np.dot(X, w1)     # N * 1
    val_1 = y1 * ss.norm().pdf(mu)
    val_2 = ss.norm().cdf(y1 * mu)
    gi = -val_1 * val_2 * X # N * D, 与书里面的符号相反，书里面写的是梯度，这里算的是导数

    return np.sum(gi, axis=0).reshape(-1, 1) + 2 * Lambda * w1  # D * 1
    
def hessian(X, y, w, Lambda):
    y1 = y.reshape(-1, 1)  # N * 1
    w1 = w.reshape(-1, 1)  # D * 1
    mu = np.dot(X, w1)     # N * 1
    pdf_mu = ss.norm().pdf(mu)
    cdf_ymu = ss.norm().cdf(y1 * mu)
    val = (pdf_mu**2 / cdf_ymu**2) + (y1 * mu * pdf_mu / cdf_ymu) # N * 1
    result = np.zeros((X.shape[1], X.shape[1]))
    for i in range(len(X)):
        xi = X[i].reshape(-1, 1)
        result += val[i] * np.dot(xi, xi.T) # 与书里面的符号相反，这里算的是二阶导数

    return result + np.diag(2 * Lambda * np.ones(X.shape[1])) # D * D

def Fit(X, y, Lambda=0.1, maxIter=50):
    w = np.zeros(X.shape[1]).reshape(-1, 1)  # init
    NLLs = []
    for i in range(maxIter):
        print('{0:-^60}'.format('Iteration: ' + str(i + 1)))
        nll = NLL(X, y, w, Lambda)
        NLLs.append(nll)
        print('NLL: ', nll)
        print('w: ', w)
        g = jac(X, y, w, Lambda)
        h = hessian(X, y, w, Lambda)
        w_next = w - np.dot(sl.inv(h), g)
        if np.allclose(w, w_next):
            break;

        w = w_next
        print('w_next: ', w)

    return w, np.array(NLLs)

w, nlls = Fit(X, y)    

        
