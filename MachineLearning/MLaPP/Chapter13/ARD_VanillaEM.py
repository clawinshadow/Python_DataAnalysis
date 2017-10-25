import numpy as np
import scipy.stats as ss
import scipy.linalg as sl
import sklearn.linear_model as slm
import sklearn.preprocessing as spp

'''
ARD: Automatic Relevance Determination, 这个还是很重要的，这个不搞清楚也没法计算RVM，基本上是同一个东西
只是RVM使用kernel扩展过的数据集而已

在大部分L1 regularization的扩展里面，对于w的概率密度都假定是factoriald的，
即 p(w) = p(w1) * p(w2) * ... * p(wd), 每一个wj都是一个Gaussian scale mixture的模型，
即 p(wj) ~ N(0, τ2), 然后 τ2 有一个先验分布，可以是很多种不同的分布，造成不同的模型
总体而言是 τ2→ wj → y ← X

但是在ARD里面，我们不再使用Factorial + GSM的模型，而是采用EB，并且将w直接积掉，只求两个
超参数α和β，分别代表wj和总体噪声的precision，然后将这两个参数plug in，回头来求E[w|τ, D]
即w的后验概率的均值，令人惊讶的是，这种方法求得的w往往是sparse的

ARD for linear regression:
    p(y|x, w, β) = N(y|w.T * x, 1/β)
    p(w) = N(w|0, A.inv), A = diag(α)
所以相比普通的Bayesian Ridge Regression，仅仅只是p(w)不一样而已，在BRR中，p(w)是isotropic的

log likelihood:
LL(α, β) = log|Cα| + y.T * Cα.inv * y, Cα = β.inv * I + X * A.inv * X.T

α, β 加上prior: IG(a, b), IG(c, d)之后的log likelihood..: 略

针对这个目标函数求解α, β后再plug in
p(w|α, β) = N(μ, Σ)
    μ = β * Σ * X.T * y
    Σ.inv = β * X.T * X + A

关键就是在于如何求 α, β
本例中的数据来源：http://scikit-learn.org/stable/auto_examples/linear_model/plot_ard.html#sphx-glr-auto-examples-linear-model-plot-ard-py

EM算法的精髓就是将 w 的 μ, Σ 看成是一个隐藏变量，在E step中求它们的期望值，再回头计算 α, β

顺便提一下 sklearn.linear_model.ARDRegression 中几个参数的含义：
alpha_1, alpha_2 : 表示 β 的prior的参数 c, d, default is 1e-6
lambda_1, lambda_2 表示的是 α 的prior的参数
threshold_lambda: 这个是与EM不同的地方，当precision大于这个参数指定的值时，会直接将对应的权重设置为0
'''

np.random.seed(0)

# prepare data
N, D = 100, 100
X = np.random.randn(N, D)
lambda_ = 4
w = np.zeros(D)
# only keep 10 weights of interest
relevant_features = np.random.randint(0, D, 10)
for i in relevant_features:
    w[i] = ss.norm.rvs(loc=0, scale=1. / np.sqrt(lambda_))
# create noise with a precision alpha of 50
alpha_ = 50
noise = ss.norm.rvs(loc=0, scale=1. / np.sqrt(alpha_), size=N)
y = np.dot(X, w) + noise

def EStep(alpha, beta, X, y):
    # calculate μ, Σ
    y = y.reshape(-1, 1)
    alpha = alpha.reshape(-1, 1)
    sigma_inv = beta * np.dot(X.T, X) + np.diag(alpha.ravel())
    sigma = sl.inv(sigma_inv)
    mu = beta * np.dot(sigma, np.dot(X.T, y))

    return mu, sigma

def MStep(mu, sigma, X, y, alpha_old, beta_old, a=1e-6, b=1e-6, c=1e-6, d=1e-6):
    N, D = X.shape
    alpha = np.zeros(D)
    s = 0
    for i in range(D):
        alpha[i] = (1 + 2*a) / (mu[i]**2 + sigma[i, i] + 2*b)
        s += 1 - alpha[i] * sigma[i, i]

    residual = np.sum((y - np.dot(X, mu))**2)
    beta = (N + 2*c) / (residual + s / beta_old + 2*d)

    return alpha, beta

def EM(X, y, fit_intercept=True, maxIter=1000):
    if fit_intercept:
        X = np.c_[np.ones(X.shape[0]), X]

    N, D = X.shape
    y = y.reshape(-1, 1)
    alpha, beta = np.ones(D), 1 / np.var(y)
    for i in range(maxIter):
        mu, sigma = EStep(alpha, beta, X, y)
        alpha_new, beta_new = MStep(mu, sigma, X, y, alpha, beta)
        mu_new, sigma_new = EStep(alpha_new, beta_new, X, y)
        print(sl.norm(alpha_new - alpha))
        if np.sum(np.abs(alpha_new - alpha)) < 1e-4:
            print('converged!')
            return alpha_new, beta_new, mu_new

        alpha = alpha_new
        beta = beta_new

    return alpha, beta, mu_new

# fit with EM
ard_EM = EM(X, y, False)
print('w by EM: \n', ard_EM[2].ravel())

# fit with sklearn
ard = slm.ARDRegression(compute_score=True, threshold_lambda=1e20, fit_intercept=False).fit(X, y)
print('intercept by sklearn: ', ard.intercept_)
print('w by sklearn: \n', ard.coef_)  # 误差基本在千分之一内
