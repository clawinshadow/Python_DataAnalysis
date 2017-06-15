import math
import numpy as np
import scipy.stats as ss
import scipy.linalg as sl
import matplotlib.pyplot as plt

'''
用贝叶斯方法来进行线性回归，可以避免MLE带来的过拟合问题，并且可以进行序列化学习
给定随机变量x1, x2, ..., xn，目标向量 tn = y(x, w) + ε， ε ~ N(0, 1/β), y(x, w) = w.T * Φ(xn)
给定先验分布 p(w) = N(w|m0, S0), 则后验分布 p(w|t) = N(w|mn, Sn), 其中：

    mn = Sn(S0.inv*m0 + β*Φ.T*t)
    Sn.inv = S0.inv + β*Φ.T*Φ

这个计算起来比较复杂，一般情况下我们给定一个各向同性的多元高斯分布就可以，即 p(w|α) = N(w|0, 1/α * I)
那么后验分布中的 mn, Sn 如下：

    mn = β*Sn*Φ.T*t
    Sn.inv = α*I + β*Φ.T*Φ

似然函数 p(t|w) = Π N(y(xn, w), 1/β)

***** 后验分布 p(t|w) 始终是关于权重向量w的函数，对它进行极大似然估计就是MAP，得到向量w(MAP) *****
*****                       因为它服从正态分布，所以w(MAP)就等于均值向量                     *****

为了方便的展示在二维图形中，下面使用一个简单的线性模型 y(x,w) = w0 + w1*x, w的先验分布就是一个二元的高斯分布
我们使用f(x, a) = -0.3 + 0.5x 来合成测试数据，xn ~ uniform(-1, 1)，噪声 ε ~ N(0, 0.2**2), 精度β相应的就等于25

给定先验分布的超参数 α = 2，那么w的先验分布就是0.5*I，然后我们通过贝叶斯方法来逐步的还原出系数-0.3和0.5
'''

def func(a0, a1, x):
    return a0 + a1 * x

def gaussian_pdf(loc, scale, x):
    return 1/(math.sqrt(2*np.pi)*scale) * np.exp(-1*np.power(x-loc, 2)/(2*scale*2))

# 似然函数与权重的后验先验无关，只跟噪声的分布有关
def likelihood(x, t, w, sigma):
    result = []                         # w 是 N * N * 2 的数组，返回结果要是 N * N 数组
    for i in range(len(w)):       
        wi = w[i]                       # 每个w[i]是 N * 2 的 二维数组
        rows = []                       # w[i]中的每一对数据计算出来的概率构成最终结果的一行
        for k in range(len(wi)):
            prob = 1                    # initial value
            weight = wi[k]
            mean = func(weight[0], weight[1], x)
            prob = prob * gaussian_pdf(mean, sigma, t)
            # prob = prob * ss.norm(mean, beta).pdf(tj) 千万不要在4万次的循环里去new一个norm的object，会非常的慢
            rows.append(prob)

        result.append(rows)

    return np.array(result)

# 根据先验概率计算后验概率，phi是design matrix，t是目标变量, 参考上面的公式
def calc_posterior(m0, s0, beta, phi, t):
    Sn_inv = sl.inv(s0) + beta * np.dot(phi.T, phi)
    Sn = sl.inv(Sn_inv)
    temp = np.dot(sl.inv(s0), m0.T)
    if temp.ndim == 1:
        temp = temp.reshape(-1, 1)
    mn = np.dot(Sn, temp + beta * np.dot(phi.T, t))

    return mn, Sn

def set_base_attrs(ax):
    ax.axis([-1, 1, -1, 1])
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])

    return ax

# ax是图形对象，rv是权重向量的概率分布
def plot_lines(ax, rv):
    ws = rv.rvs(6)
    x = np.linspace(-1, 1, 200)
    for i in range(len(ws)):
        w = ws[i]
        y = func(w[0], w[1], x)
        ax.plot(x, y, 'r-')

    return ax

# uniform 的两个参数loc, scale, 决定了样本的范围是(loc, loc+scale)
rv= ss.uniform(-1, 2)
x = rv.rvs(20, np.random.RandomState(seed=2100))
a0, a1 = -0.3, 0.5
t = func(a0, a1, x)

sigma = 0.2
beta = 1/sigma**2
rv2 = ss.norm(0, sigma)    # distribution of noise
t_n = t + rv2.rvs(20)     # synthetic data with noise additive

alpha = 2
mean = np.array([0, 0])
cov = 1/alpha * np.eye(2)
rv3 = ss.multivariate_normal(mean, cov)  # 权重向量w的先验分布

fig = plt.figure(figsize=(8, 9.5))
fig.canvas.set_window_title('Bayesian Linear Regression')
ax = plt.subplot(431)
ax.set_visible(False)

ax2 = plt.subplot(432, aspect='equal')   # 保证图形是个正方形
ax2 = set_base_attrs(ax2)
x2, y2 = np.mgrid[-1:1:0.01, -1:1:0.01]
pos = np.dstack((x2, y2))
probs = rv3.pdf(pos)
levels = np.linspace(probs.min(), probs.max(), 200)
ax2.set_title('prior/posterior')
ax2.contourf(x2, y2, probs, levels, cmap='jet')

ax3 = plt.subplot(433, aspect='equal')
ax3 = set_base_attrs(ax3)
ax3 = plot_lines(ax3, rv3)
ax3.set_title('data space')

ax4 = plt.subplot(434, aspect='equal')
ax4 = set_base_attrs(ax4)
ax4.set_title('likelihood')
x4 = x[1]
t4 = t[1]
probs4 = likelihood(x4, t4, pos, sigma)
levels4 = np.linspace(probs4.min(), probs4.max(), 200)
ax4.contourf(x2, y2, probs4, levels4, cmap='jet')
ax4.plot(-0.3, 0.5, 'w+')

ax5 = plt.subplot(435, aspect='equal')
ax5 = set_base_attrs(ax5)
phi = np.c_[np.ones(1).reshape(-1, 1), x4.reshape(-1, 1)]
t5 = t4.reshape(-1, 1)
posterior_mean, posterior_cov = calc_posterior(mean, cov, beta, phi, t5)
rv5 = ss.multivariate_normal(posterior_mean.ravel(), posterior_cov)
probs5 = rv5.pdf(pos)
levels5 = np.linspace(probs5.min(), probs5.max(), 200)
ax5.contourf(x2, y2, probs5, levels5, cmap='jet')
ax5.plot(-0.3, 0.5, 'w+')

ax6 = plt.subplot(436, aspect='equal')
ax6 = set_base_attrs(ax6)
ax6 = plot_lines(ax6, rv5)
ax6.plot(x4, t4, 'bo')

ax7 = plt.subplot(437, aspect='equal')
ax7 = set_base_attrs(ax7)
x7 = x[2]
t7 = t[2]
probs7 = likelihood(x7, t7, pos, sigma)
levels7 = np.linspace(probs7.min(), probs7.max(), 200)
ax7.contourf(x2, y2, probs7, levels7, cmap='jet')
ax7.plot(-0.3, 0.5, 'w+')

ax8 = plt.subplot(438, aspect='equal')
ax8 = set_base_attrs(ax8)
phi8 = np.c_[np.ones(1).reshape(-1, 1), x7.reshape(-1, 1)]
t8 = t7.reshape(-1, 1)
posterior_mean8, posterior_cov8 = calc_posterior(posterior_mean.ravel(), posterior_cov, beta, phi8, t8)
rv8 = ss.multivariate_normal(posterior_mean8.ravel(), posterior_cov8)
probs8 = rv8.pdf(pos)
levels8 = np.linspace(probs8.min(), probs8.max(), 200)
ax8.contourf(x2, y2, probs8, levels8, cmap='jet')
ax8.plot(-0.3, 0.5, 'w+')

ax9 = plt.subplot(439, aspect='equal')
ax9 = set_base_attrs(ax9)
ax9 = plot_lines(ax9, rv8)
ax9.plot([x4, x7], [t4, t7], 'bo')

ax10 = plt.subplot(4, 3, 10, aspect='equal')
ax10 = set_base_attrs(ax10)
x10 = np.array(x[-1])
t10 = np.array(t[-1])
print(x10, t10)
probs10 = likelihood(x10, t10, pos, sigma)
levels10 = np.linspace(probs10.min(), probs10.max(), 200)
ax10.contourf(x2, y2, probs10, levels10, cmap='jet')
ax10.plot(-0.3, 0.5, 'w+')

ax11 = plt.subplot(4, 3, 11, aspect='equal')
ax11 = set_base_attrs(ax11)
xs = x[2:]
ts = t[2:]
phi11 = np.c_[np.ones(18).reshape(-1, 1), xs.reshape(-1, 1)]
t11 = ts.reshape(-1, 1)
posterior_mean_final, posterior_cov_final = calc_posterior(posterior_mean8.ravel(), posterior_cov8, beta, phi11, t11)
rv11 = ss.multivariate_normal(posterior_mean_final.ravel(), posterior_cov_final)
probs11 = rv11.pdf(pos)
levels11 = np.linspace(probs11.min(), probs11.max(), 200)
ax11.contourf(x2, y2, probs11, levels11, cmap='jet')
ax11.plot(-0.3, 0.5, 'w+')

ax12 = plt.subplot(4, 3, 12, aspect='equal')
ax12 = set_base_attrs(ax12)
ax12 = plot_lines(ax12, rv11)
ax12.plot(x, t, 'bo')

plt.show()
