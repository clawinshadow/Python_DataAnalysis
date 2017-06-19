import numpy as np
import scipy.stats as ss
import scipy.linalg as sl
import sklearn.gaussian_process as sgp
import sklearn.gaussian_process.kernels as kernels
import matplotlib.pyplot as plt

'''
画一下高斯过程对应的先验分布和后验分布，kernel以RBF为例，我们已经知道先验分布其实就是：

    f(x*) ~ GP(0, K(X*, X*))

按道理来说，高斯过程是对一个连续域的分布，要画图形的话x轴上应该是无穷多个数据点，但我们给定一个区间，再取几百个数据点就够了
其中X*是样本点，K是核函数计算出来的Gram矩阵，如果保持样本点集X*一致，那么决定这个先验分布不同的唯一之处就在于选取不同的核函
数了，其实先验分布只由核函数唯一确定，均值一般都默认为0了，均值不影响什么，然后呢样本点集X*只是我们为了画图方便构造出来的点
集，一百个两百个其实区别不大，只要图形看起来是连续的就行，那么如何去画GPML和PRML中都出现过的图呢：

1. 选取某个区间，均匀采样数百个点集，用于保证图形是光滑的
2. 选取某个核函数，然后计算出对应的Gram矩阵，矩阵规模决定于步骤1中样本点集的个数
3. 这时候已经生成了f*的概率分布N(0, K(X*, X*)), 然后用它来生成样本点向量f*，每采样一次，就生成一个与步骤1中的X树木一致的f*
   向量，那么就可以画出一条独立的曲线，采样10次就是10条曲线

'''

def f(x):
    return np.sin(x-2)*x

def rbf_kernel(xn, xm):
    return np.exp(-0.5 * sl.norm(xn - xm)**2)

def gram(x):
    result = []
    for i in range(len(x)):
        rows = []
        xi = x[i]
        for j in range(len(x)):
            xj = x[j]
            rows.append(rbf_kernel(xi, xj))
        result.append(rows)

    return np.array(result)

# use sklean
# 默认参数下的RBF等价于np.exp(-0.5 * sl.norm(xn, xm)**2)
k = kernels.RBF()
gp = sgp.GaussianProcessRegressor(kernel=k)  # 本例中采用高斯过程来解决曲线拟合和对应的回归问题，所以用Regressor
X_ = np.linspace(0, 5, 200)                  # X*
y_samples = gp.sample_y(X_.reshape(-1, 1), 10, random_state=np.random.seed(1))
print(y_samples.shape)

# do it myself
K = gram(X_)
mean = np.tile(0, len(X_))
prior = ss.multivariate_normal(mean, K, allow_singular=True)
y_samples2 = prior.rvs(size=10, random_state=np.random.seed(1)) # 一样的随机数，画出来的图形会与上面的一模一样
print(y_samples2.shape)

fig = plt.figure(figsize=(12, 10))
fig.canvas.set_window_title('Gaussian Process Prior&Posterior')
plt.subplot(221)
plt.plot(X_, y_samples, lw=1)
plt.title('Prior by sklearn')
plt.axis([0, 5, -3, 3])

plt.subplot(222)
plt.plot(X_, y_samples2.T, lw=1)
plt.title('Prior by myself')
plt.axis([0, 5, -3, 3])

# 假设真正的f(x)=x*sin(x-2), 生成五个数据点，然后据此求X_的后验概率P(X_|obs_X)
# 使用高斯过程，RBF核函数来拟合真正的f(x)
obs_X = np.linspace(0.5, 4.8, 5)
obs_Y = f(obs_X)

data = np.r_[obs_X, X_]        # 将训练数据和测试数据连接在一起
gram_all = gram(data)          # 生成所有数据联合Gram矩阵，205 * 205
K_aa = gram_all[:5, :5]        # 对应先验的矩阵 5 * 5
K_ab = gram_all[:5, 5:]        # 5 * 200
K_ba = gram_all[5:, :5]        # 200 * 5
K_bb = gram_all[5:, 5:]        # 200 * 200

mean_posterior = np.dot(np.dot(K_ba, sl.inv(K_aa)), obs_Y)
cov_posterior = K_bb - np.dot(np.dot(K_ba, sl.inv(K_aa)), K_ab)
posterior = ss.multivariate_normal(mean_posterior, cov_posterior, allow_singular=True)
y_samples3 = posterior.rvs(size=4, random_state=np.random.seed(1))

plt.subplot(223)
plt.plot(obs_X, obs_Y, 'b+', markersize=10)
plt.plot(X_, y_samples3.T, lw=1)
plt.plot(X_, f(X_), 'y--', label='Real Func')
plt.legend()
plt.title('Posterior by my self')
plt.axis([0, 5, y_samples3.min()-1, y_samples3.max()+1])

# use sklearn to fit & predict
gp.fit(obs_X.reshape(-1, 1), obs_Y.reshape(-1, 1))
# return_std 和 return_cov 只能有一个为True，std是计算每个数据点的标准差，cov是计算所有点组合起来的协方差阵
y_mean, y_std = gp.predict(X_.reshape(-1, 1), return_std=True)

plt.subplot(224)
plt.plot(obs_X, obs_Y, 'b+', markersize=10)
plt.plot(X_, y_mean, 'g-')
plt.plot(X_, f(X_), 'y--')
plt.title('Posterior by sklearn')
plt.fill_between(X_, y_mean.ravel() - y_std, y_mean.ravel() + y_std, alpha=0.2, color='r')

plt.show()
