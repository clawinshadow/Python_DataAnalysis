import numpy as np
import scipy.io as sio
import scipy.stats as ss
import scipy.linalg as sl
import sklearn.linear_model as slm
import matplotlib.pyplot as plt

'''大概要跑十几秒钟，吉布斯采样果然是慢'''

np.random.seed(0)

# load data
data = sio.loadmat('mathDataHoff.mat')
print(data.keys())
y = data['y']
schools = y[:, 0]
school_ids = np.unique(schools)
score = y[:, 3]
ses = y[:, 2]

# for each school, fit with OLS individually, and calculate temp datas
Ws = np.zeros((len(school_ids), 2))
sampleSize = np.zeros(len(school_ids))
sigmahat = np.zeros((len(school_ids)))
XXs = np.zeros((len(school_ids), 2, 2))
Xys = np.zeros((len(school_ids), 2))
ys = []
xs = []
for i in range(len(school_ids)):
    school = school_ids[i]
    idx = schools == school
    yi = score[idx].reshape(-1, 1)
    xi = ses[idx]
    xi = xi - np.mean(xi)  # center x
    X = np.c_[np.ones(len(xi)), xi]

    # calculating temp datas
    XX = np.dot(X.T, X)
    Xy = np.dot(X.T, yi)

    # fit with OLS
    wi = np.dot(sl.inv(XX), Xy)
    Ws[i] = wi.ravel()  # first is intercept, second is slope

    sampleSize[i] = len(xi)
    XXs[i] = XX
    Xys[i] = Xy.ravel()
    ys.append(yi.ravel())
    xs.append(X)
    sigmahat[i] = np.var((yi - np.dot(X, wi)), ddof=1) # unbiased variance

w_avg = np.mean(Ws, axis=0)
xvals = np.linspace(-3, 3, 200)
xvalsi = np.c_[np.ones(len(xvals)), xvals]

# plot original data with OLS fitting
fig = plt.figure(figsize=(13, 4))
fig.canvas.set_window_title('multilevelLinregDemo')

ax = plt.subplot(131)
ax.tick_params(direction='in')
plt.axis([-3, 3, 20, 85])
plt.xticks(np.linspace(-2, 2, 5))
plt.yticks(np.linspace(20, 80, 7))
plt.xlabel('SES')
plt.ylabel('Math Score')
for i in range(len(Ws)):
    wi = Ws[i].reshape(-1, 1)
    yvals = np.dot(xvalsi, wi).ravel()
    plt.plot(xvals, yvals, '-', color=[0.7, 0.7, 0.7], lw=2)
yvals = np.dot(xvalsi, w_avg.reshape(-1, 1))
plt.plot(xvals, yvals, 'k-', lw=3)

ax = plt.subplot(132)
ax.tick_params(direction='in')
plt.axis([0, 35, -8, 12])
plt.xticks(np.linspace(5, 30, 6))
plt.yticks(np.linspace(-5, 10, 4))
plt.xlabel('Sample Size')
plt.ylabel('Slope')
plt.plot(sampleSize, Ws[:, 1], 'o', linestyle='none', fillstyle='none')
plt.axhline(w_avg[1], color='k', lw=3)

# Fit with Gibbs Sampling
p = 2
J = len(school_ids)
N = len(y)

# prior for μw ∼ N(μ0, V0), mean of overall coef_ w
mu0 = w_avg
V0 = np.cov(Ws.T)   # 注意numpy.cov()里面传进去的矩阵，视每一行为一个单元，不是每一列
invV0 = sl.inv(V0)  # 避免后面重复计算

# prior for Σw ∼ IW(η0 ,sl.inv(S0)), covariance of overall coef_ w
S0 = V0
eta0 = p + 2

# prior for σ2 ∼ IG(ν0/2, ν0*σ0**2/2), noise sigma
nu0 = 1
sigma02 = np.mean(sigmahat)

# initialize
muw = ss.multivariate_normal(mean=mu0, cov=V0).rvs(1)
# # 书中的是inverse-wishart分布，这里用wishart分布，得出来的抽样就是前者的inv，避免后面再每次重复计算矩阵的逆
invSigma = ss.wishart(df=eta0, scale=sl.inv(S0)).rvs(1)
# 这个与书里面的表示形式也不一样，一个是gamma，一个是inverse gamma，但实际效果是一样的
a = 0.5 * (nu0 + N)
b = 2 * (nu0 * sigma02)
sigma2 = 1 / ss.gamma.rvs(a, scale=b, size=1)

# gibbs sampler
Ws_sample = np.zeros(Ws.shape)
muw_sample = []
Wss = []
for i in range(500):
    ssr = 0
    muw = muw.reshape(-1, 1)
    # sample p(wj|Dj, θ), θ including {μw, Σw, σ2}, because θ is overall parameters for the whole dataset,
    # so it's the meaning of small sample-size data borrows statical strength from the big ones
    for j in range(J):
        Vj = sl.inv(invSigma + XXs[j] / sigma2)
        muj = np.dot(Vj, np.dot(invSigma, muw) + Xys[j].reshape(-1, 1) / sigma2)
        Ws_sample[j] = ss.multivariate_normal(muj.ravel(), Vj).rvs(1)   # update w[1:j]
        ssr += np.sum((ys[j] - np.dot(xs[j], Ws_sample[j].reshape(-1, 1)).ravel())**2)  # sum square of residuals

    # sample p(μw|w1:J,Σw)
    sigmaN = sl.inv(invV0 + J * invSigma)
    muN = np.dot(sigmaN, np.dot(invV0, mu0.reshape(-1, 1)) + J * np.dot(invSigma, np.mean(Ws_sample, axis=0).reshape(-1, 1)))
    muw = ss.multivariate_normal(muN.ravel(), sigmaN).rvs(1)  # update  μw

    # sample p(Σw|μw,w1:J)
    eta = eta0 + J
    diff = Ws_sample - muw
    diff1 = diff.reshape(diff.shape[0], diff.shape[1], 1)
    diff2 = diff.reshape(diff.shape[0], 1, diff.shape[1])
    Smu = np.sum(diff1 * diff2, axis=0)
    invS = sl.inv(S0 + Smu)
    invSigma = ss.wishart(df=eta, scale=invS).rvs(1)  # update Σw

    # sample p(σ2|D,w1:J)
    a = 0.5 * (nu0 + N)
    b = 1 / (0.5 * (nu0 * sigma02 + ssr))
    sigma2 = 1 / ss.gamma.rvs(a, scale=b, size=1)   # update σ2

    if i % 10 == 9:
        Wss.append(Ws_sample)

Wss = np.array(Wss)
print(Wss.shape)   # should be 50 * 100 * 2
ws_gibbs_sampling = np.mean(Wss, axis=0)
w_avg_gs = np.mean(ws_gibbs_sampling, axis=0)

ax = plt.subplot(133)
ax.tick_params(direction='in')
plt.axis([-3, 3, 20, 85])
plt.xticks(np.linspace(-2, 2, 5))
plt.yticks(np.linspace(20, 80, 7))
plt.xlabel('SES')
plt.ylabel('Math Score')
for i in range(len(ws_gibbs_sampling)):
    wi = ws_gibbs_sampling[i].reshape(-1, 1)
    yvals = np.dot(xvalsi, wi).ravel()
    plt.plot(xvals, yvals, '-', color=[0.7, 0.7, 0.7], lw=2)
yvals = np.dot(xvalsi, w_avg_gs.reshape(-1, 1))
plt.plot(xvals, yvals, 'k-', lw=3)

plt.tight_layout()
plt.show()