import pprint
import numpy as np
import scipy.io as sio
import scipy.stats as ss
import scipy.linalg as sl
import sklearn.preprocessing as spp
import matplotlib.pyplot as plt
import matplotlib.animation as ma
from mixGaussFitVBEM import *

'''
代码好写，主要是里面的思想和本质，为什么我们要在普通EM的基础上再引入VBEM？

首先，我们回顾下带隐藏变量Z的模型，以mix gaussian为例，简单点就是 zi -> xi <- θ, 隐藏变量z随着x的数量增长而增长，
有多少个xi就有多少个zi，而θ独立于数据集的数量，只跟cluster数量K有关。

所以普通EM将隐藏变量Z和模型参数θ区别对待，如下：
    - 在E步中，它根据xi和θ_old来计算隐藏变量当前的后验概率分布 P(zi|xi, θ_old；
    - 然后在M步中再更新θ的MAP，得到θ_new
    
而在VBEM中，我们将隐藏变量zi视为和θ一样的地位，都是模型参数，那么问题又来了，既然这样我们为什么还需要分EM两步来求解呢？
先看一下VBEM的定义，它依然是用mean field的思想来给后验分布建模，即：

    p(θ, z1:N |D) ≈ q(θ)q(z) = q(θ)*Π(q(zi))
    
第一个分解，即p(θ, z1:N |D) ≈ q(θ)q(z)，是这个算法可行最重要的一个假设，从这个定义可以很明显地看到，我们计算的后验分布
既包括了模型参数θ也包括了隐藏变量z1:N，所以VBEM会更bayes一点，按我的观点算是一个full bayes的模型，VBEM中的EM两步如下：
    - E步中，更新所有的q(zi|D)，而在普通EM的E步中，它是将上一次θ的MAP plugg in进来
    - M步中，更新q(θ|D)，在普通EM的M步中，它是计算一个MAP，而在这里，它是更新所有的hyper parameters
    
除此之外，VBEM相比EM最大的一个好处就是它可以通过计算Marginal Likelihood的下界，从而进行model selection，这个是普通EM
所不具备的，因为普通EM本质上是用来解决MLE的问题，它是一个frequentist的方法，而不是bayesian的方法

关于VBEM，在PRML一书中讲的要更好，更详细，参见Page 474 in PRML
'''

def parseModel(struct):
    alpha = struct[0].reshape(-1, 1)
    beta = struct[1].reshape(-1, 1)
    m = struct[2]
    v = struct[3].reshape(-1, 1)
    W = struct[4]
    D1, D2, K = W.shape
    W = W.ravel().reshape(K, D1, D2, order='F')

    return mixGaussBayesStructure(alpha, beta, m, v, W)

def draw(ax, X, params):
    points = ax.plot(X[:, 0], X[:, 1], 'o', linestyle='none', mew=1, fillstyle='none')
    center = ax.plot([0], [0], 'ro', fillstyle='none')
    alpha = params['alpha']
    m = params['m']
    v = params['v']
    W = params['W']
    K, D = m.shape
    weights = alpha / np.sum(alpha)
    xx, yy = np.meshgrid(np.linspace(-2, 1.5, 200), np.linspace(-2.5, 2, 200))
    xs = np.c_[xx.ravel(), yy.ravel()]

    artists = [points, center]
    for i in range(K):
        if weights[i] < 0.001:
            continue
        cov = sl.inv(W[i]) / (v[i] - D - 1)
        zz = ss.multivariate_normal(m[i], cov).pdf(xs).reshape(xx.shape)
        level = zz.min() + 0.5 * (zz.max() - zz.min())
        artists.append(ax.contourf(xx, yy, zz, colors='gray', levels=[level, zz.max()]))
        x = np.asscalar(m[i, 0] + 0.02)
        y = np.asscalar(m[i, 1] + 0.02)
        artists.append(ax.text(x, y, str(i+1), bbox=dict(facecolor='yellow', alpha=0.5)))

    return artists

def draw2(ax, lowerbounds):
    N = len(lowerbounds)
    return ax.plot(np.linspace(1, N, N), lowerbounds, 'o-', color='midnightblue', lw=1, fillstyle='none')

def draw3(ax, params):
    alpha = params['alpha'].ravel()
    K = len(alpha)
    return ax.bar(np.linspace(1, K, K), alpha, color='midnightblue', edgecolor='none', align='center')

class SeqUpdate(object):
    def __init__(self, ax1, ax2, ax3, X, params, lbs):
        self.params = params
        self.X = X
        self.lbs = lbs

        self.ax1 = ax1
        self.ax1.set_xlim(-2, 1.5)
        self.ax1.set_ylim(-2.5, 2)
        self.ax1.set_xticks(np.linspace(-2, 1.5, 8))
        self.ax1.set_yticks(np.linspace(-2.5, 2, 10))
        self.ax1.tick_params(direction='in')

        self.ax2 = ax2
        self.ax2.set_title('variational Bayes objective for GMM on old faithful data', fontdict={'fontsize': 10})
        self.ax2.set_ylabel('ower bound on log marginal likelihood')
        self.ax2.set_xlabel('iter')
        self.ax2.set_xlim(0, 100)
        self.ax2.set_ylim(-1100, -600)
        self.ax2.set_xticks(np.linspace(0, 100, 6))
        self.ax2.set_yticks(np.linspace(-1100, -600, 11))
        self.ax2.tick_params(direction='in')

        self.ax3 = ax3
        self.ax3.set_xlim(0, 7)
        self.ax3.set_xticks(np.linspace(0, 6, 7))
        self.ax3.tick_params(direction='in')

    def __call__(self, i):
        self.ax1.clear()
        self.ax3.clear()
        self.ax1.set_title('Iteration: {0}'.format(i + 1))
        self.ax3.set_title('Iteration: {0}'.format(i + 1))

        artists = draw(self.ax1, self.X, params[i])
        artists.append(draw2(self.ax2, self.lbs[:i+1]))
        artists.append(draw3(self.ax3, params[i]))

        return artists

# load data
data = sio.loadmat('faithful.mat')
print(data.keys())
X = data['faithful']
X = spp.StandardScaler().fit_transform(X)
N, D = X.shape
X = X * np.sqrt((N - 1) / N)  # 修正sklearn与matlab codes里面关于standardization的误差
# print(X)
maxIter = 200

model = sio.loadmat('model.mat')  # initial value
model = model['model']
priorParams = parseModel(model[0][0][0][0][0])
postParams = parseModel(model[0][0][2][0][0])
# pprint.pprint(priorParams)
# pprint(postParams)

# Fit with VariationalBayesEM
res = VBEM(priorParams, postParams, X, maxIter)
print('trace of lowerbounds: \n', res[1])
print('converged params: \n')
pprint.pprint(res[0][-1])
params = res[0]
lowerbounds = res[1]

# plots
fig = plt.figure(figsize=(16, 5))
fig.canvas.set_window_title('mixGaussVbDemoFaithful')

ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)

su = SeqUpdate(ax1, ax2, ax3, X, params, lowerbounds)
anim = ma.FuncAnimation(fig, su, frames=len(params), interval=500, repeat=False)

plt.tight_layout()
plt.show()
