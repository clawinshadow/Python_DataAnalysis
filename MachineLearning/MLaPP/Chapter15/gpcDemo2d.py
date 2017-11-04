import numpy as np
import scipy.io as sio
import scipy.linalg as sl
import scipy.special as ss
import scipy.stats as stats
import sklearn.gaussian_process as sgp
import sklearn.metrics.pairwise as smp
import sklearn.gaussian_process.kernels as sk
import matplotlib.pyplot as plt

'''
demos about gaussian process classification, with logistic regression
pay attention to the below gradient & hessian of logistic & probit regression model

original Log likelihood log p(yi|fi)   |       gradient         |        Hessian
          log sigmoid(yi * fi)                  ti − πi                −πi(1 − πi)
          log Φ(yi * fi)                  yi * φ(fi) / Φ(yi* fi)     ...............

yi ∈ {-1, +1}, ti = (yi + 1) / 2 ∈ {0, 1},
πi = sigm(fi) for logistic regression, and πi = Φ(fi) for probit regression
φ and Φ are the pdf and cdf of N(0, 1).

details refer to P.525 in the book
Algorithm refer to P.529, using probit regression in this demo
'''

# load data
data = sio.loadmat('gpcDemo2d.mat')
print(data.keys())
x = data['x']
y = data['y']
print(x.shape, y.shape)

# Fit with GPC(Logistic regression)
def se_kernelise(X, Y, v_scale, h_scale):
    gamma = 1 / (2 * h_scale**2)
    K = smp.rbf_kernel(X, Y, gamma=gamma)

    return (v_scale**2) * K

# 本例中并未用到， 这个方法主要是后面的p(y=1|x)不太好求解
def IRLS_LogisticRegression(K, Y):
    N = len(K)
    f = np.zeros((N, 1))
    maxIter = 1000
    t = (Y + 1) / 2                  # N * 1, {0, 1}
    for i in range(maxIter):
        pi = 1 / (1 + np.exp(-1 * f))         # N * 1,
        g = t - pi                            # N * 1, L1 derivative
        H = np.diag((pi * (pi - 1)).ravel())  # N * N, L2 derivative
        W = -1 * H

        B = np.eye(N) + np.dot(np.sqrt(W), np.dot(K, np.sqrt(W)))  # N * N
        L = sl.cholesky(B, lower=True)            # N * N
        b = np.dot(W, f) + g                      # N * 1
        val1 = np.dot(np.sqrt(W), np.dot(K, b))   # N * 1
        val2 = np.dot(sl.inv(L.T), sl.inv(L))     # N * N
        a = b - np.dot(np.sqrt(W), np.dot(val2, val1)) # N * 1
        f_new = np.dot(K, a)                      # N * 1

        if np.allclose(f_new, f):
            f = f_new
            break

        f = f_new

    # calculate Log Likelihood of training set
    pyf = np.sum(1 / (1 + np.exp(-1 * Y * f)))  # N * 1

    return f, g, W, a, L, pyf

# 如果直接求导的话会有一部分二阶导数的值是负数。。这样后面开方的时候就没法计算了
def Get_NP(Y, f):
    yf = (Y * f).ravel()
    N = len(f)
    p = (1 + ss.erf(yf / np.sqrt(2))) / 2  # (N, )
    lp = np.zeros(N)
    b = 0.158482605320942
    c = -1.785873318175113
    ok = yf > -6
    lp[ok] = np.log(p[ok])
    lp[~ok] = -1 * yf[~ok]**2 / 2 + b * yf[~ok] + c

    n_p = np.zeros(N)
    ok = yf > -5
    n_p[ok] = (np.exp(-1 * yf[ok]**2 / 2) / np.sqrt(2 * np.pi)) / p[ok]

    bd = yf < -6
    n_p[bd] = np.sqrt(1 + (yf[bd]**2) / 4) - yf[bd] / 2

    interp = ~ok & ~bd
    tmp = yf[interp]
    lam = -5 - yf[interp]
    n_p[interp] = (1 - lam) * ((np.exp(-1 * tmp**2 / 2) / np.sqrt(2 * np.pi)) / p[interp]) +\
        lam * (np.sqrt(1 + (tmp**2) / 4) - tmp / 2)

    out1 = np.sum(lp)                       # log p(Y|f)
    out2 = Y * (n_p.reshape(-1, 1))         # gradient vector
    out3 = np.diag(-1 * n_p**2 - yf * n_p)  # Hessian matrix

    return out1, out2, out3

def Probs(mu, var):
    mu = mu.ravel()
    var = np.diag(var)
    z = mu / np.sqrt(1 + var)
    probs = (1 + ss.erf(z /np.sqrt(2))) / 2

    return probs

def IRLS_ProbitRegression(K, Y):
    N = len(K)
    f = np.zeros((N, 1))
    maxIter = 1000
    t = stats.norm().cdf(Y * f)
    psi = stats.norm().pdf(f)
    for i in range(maxIter):
        lp, g, H = Get_NP(Y, f)
        W = -1 * H  # L2 derivative should be N * N

        B = np.eye(N) + np.dot(np.sqrt(W), np.dot(K, np.sqrt(W)))  # N * N
        L = sl.cholesky(B, lower=True)  # N * N
        b = np.dot(W, f) + g  # N * 1
        val1 = np.dot(np.sqrt(W), np.dot(K, b))  # N * 1
        val2 = np.dot(sl.inv(L.T), sl.inv(L))  # N * N
        a = b - np.dot(np.sqrt(W), np.dot(val2, val1))  # N * 1
        f_new = np.dot(K, a)  # N * 1

        if np.allclose(f_new, f):
            f = f_new
            break

        f = f_new

    return f, g, W, a, L, np.sum(np.log(t))


def GPC(X, Y, v_scale, h_scale, xtest, method):
    K = se_kernelise(X, X, v_scale, h_scale)   # N * N

    # Calculate f
    if method == 'logistic':
        f, g, W, a, L, pyf = IRLS_LogisticRegression(K, Y)
    if method == 'probit':
        f, g, W, a, L, pyf = IRLS_ProbitRegression(K, Y)
    else:
        raise ValueError('Unknown Method')

    # calculate Log Likelihood of training set
    LL = pyf - 0.5 * np.dot(a.T, f) - np.sum(np.log(np.diag(L)))

    # predict test set
    Ks = se_kernelise(X, xtest, v_scale, h_scale)  # N * Ns
    Kss = se_kernelise(xtest, xtest, v_scale, h_scale)
    mu = np.dot(Ks.T, g)  # Ns * 1
    v = np.dot(sl.inv(L), np.dot(np.sqrt(W), Ks))  # N * Ns
    var = Kss - np.dot(v.T, v)  # Ns * Ns

    probs = Probs(mu, var)

    return f, mu, var, probs

xx, yy = np.meshgrid(np.linspace(-4, 4, 81), np.linspace(-4, 4, 81))
xtest = np.c_[xx.ravel(order='F'), yy.ravel(order='F')] # keep the same sequence as in matlab code

f, mu, var, probs = GPC(x, y, 10, 0.5, xtest, 'probit')
probs = np.round(probs, 4)  # force some probability equal to 0.5000
# print(f.ravel())    # absolute match with in matlab
# print(probs[0:50])  # absolute match with in matlab

# Fit with sklearn.GPC
# Attention: 1. currently GPC is restriced to implement by logistic link function,
#               so the predict results is not the same as learning hyperparams by probit link
#            2. GPC在计算的过程中会自动优化kernel的超参数，但是kernel 如何定义很重要，
#               下面代码中所示的定义代表了GPC会自动优化RFB的两个参数，即v_scale, h_scale
#               如果定义 kernel = sk.RBF(length_scale=1.0), 则只会优化h_scale
#            3. Fit完毕后，可以调用GPC.kernel_.theta来获取优化过的超参数，但是要exp一下
kernel = 1.0 * sk.RBF(length_scale=1.0)
GPC = sgp.GaussianProcessClassifier(kernel=kernel).fit(x, y.ravel())
probs_sklearn = GPC.predict_proba(xtest)[:, 1]  # 两个分类的预测概率都有
print(GPC.kernel_)
print(np.exp(GPC.kernel_.theta))
print(xtest.shape, probs_sklearn.shape)
v_scale = np.sqrt(np.exp(GPC.kernel_.theta)[0])
h_scale = np.exp(GPC.kernel_.theta)[1]
optimalStr = r'$SE kernel, l={0:.4f}, \sigma={1:.4f}$'.format(h_scale, v_scale)

# plots
fig = plt.figure(figsize=(11, 5))
fig.canvas.set_window_title('gpcDemo2d')

def plot(index, title, probs_):
    plt.subplot(index)
    plt.title(title)
    plt.axis([-4, 4, -4, 4])
    plt.xticks(np.linspace(-4, 4, 9))
    plt.yticks(np.linspace(-4, 4, 9))
    index = y.ravel() == -1
    probs_ = probs_.reshape(xx.shape, order='F')
    plt.plot(x[index][:, 0], x[index][:, 1], 'b+', ms=10, linestyle='none')
    plt.plot(x[~index][:, 0], x[~index][:, 1], 'ro', ms=10, linestyle='none', fillstyle='none')
    plt.contour(xx, yy, probs_, np.linspace(0.1, 0.9, 9), cmap='jet')
    plt.contour(xx, yy, probs_, levels=[0.5], color='k', lw=2)

    cax = plt.imshow(probs_, interpolation='nearest', cmap='jet')
    cax.set_visible(False)
    fig.colorbar(cax, ticks=np.linspace(0.1, 0.9, 9))

# 书里面的sigma2是错的，实际上只是个标准差而已，不是方差
plot(121, r'$SE kernel, l=0.500, \sigma=10.000$', probs)
plot(122, optimalStr, probs_sklearn)  # 图形几乎是一样的，只不过参数不一样，是因为link func不一样

plt.tight_layout()
plt.show()