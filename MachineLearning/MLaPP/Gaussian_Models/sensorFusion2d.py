import numpy as np
import scipy.stats as ss
import scipy.linalg as sl
import matplotlib.pyplot as plt

# 与gaussInferParamsMean2d中不同的是，每个观测点y的噪声是不一样的，即协方差矩阵不一样
def Draw(index, var_1, var_2, xlim, ylim):
    prior_mu = np.array([0, 0])
    prior_cov = 10**10 * np.eye(2)

    x1 = np.array([0, -1])
    x2 = np.array([1, 0])
    avg = np.array([0.5, -0.5])
    # x3 = np.array([0.5, -0.5])

    # 这两个点要分开计算，一次算一个，后验概率再作为下一个的先验概率
    post_cov_1_inv = sl.inv(prior_cov) + sl.inv(var_1)
    post_mu_1 = np.dot(sl.inv(post_cov_1_inv), np.dot(sl.inv(var_1), x1.reshape(-1, 1)))

    post_cov_2_inv = post_cov_1_inv + sl.inv(var_2)
    post_mu_2 = np.dot(sl.inv(post_cov_2_inv), np.dot(sl.inv(var_2), x2.reshape(-1, 1)) +\
                       np.dot(post_cov_1_inv, post_mu_1.reshape(-1, 1)))

    rv1 = ss.multivariate_normal(x1, var_1)
    rv2 = ss.multivariate_normal(x2, var_2)
    rv3 = ss.multivariate_normal(post_mu_2.ravel(), sl.inv(post_cov_2_inv))

    x, y = np.mgrid[xlim[0]:xlim[1]:0.01, ylim[0]:ylim[1]:0.01]
    z = np.dstack((x, y))
    probs_1 = rv1.pdf(z)
    probs_2 = rv2.pdf(z)
    probs_3 = rv3.pdf(z)

    # 圆里面囊括80%的概率密度
    levels_1 = probs_1.min() + 0.2 * (probs_1.max() - probs_1.min())
    levels_2 = probs_2.min() + 0.2 * (probs_2.max() - probs_2.min())
    levels_3 = probs_3.min() + 0.2 * (probs_3.max() - probs_3.min())
    
    plt.subplot(index)
    plt.plot(x1[0], x1[1], 'rx')
    plt.plot(x2[0], x2[1], 'gx')
    plt.plot(post_mu_2.ravel()[0], post_mu_2.ravel()[1], 'kx')
    plt.contour(x, y, probs_1, levels=[levels_1], colors='red')
    plt.contour(x, y, probs_2, levels=[levels_2], colors='green')
    plt.contour(x, y, probs_3, levels=[levels_3], colors='black')
    plt.xlim(xlim)
    plt.ylim(ylim)

var_1 = 0.01 * np.eye(2)
var_2 = 0.01 * np.eye(2)
xlim = [-0.4, 1.4]
ylim = [-1.4, 0.4]
    
plt.figure(figsize=(14, 4))
Draw(131, var_1, var_2, xlim, ylim)

var_1 = 0.05 * np.eye(2)  # a weak sensor, posterior 会更靠近另一个观测值
xlim = [-0.5, 1.4]
Draw(132, var_1, var_2, xlim, ylim)

var_1 = 0.01 * np.array([[10, 1],
                         [1, 1]])
var_2 = 0.01 * np.array([[1, 1],
                         [1, 10]])
xlim = [-1, 1.5]
ylim = [-1.5, 1]
Draw(133, var_1, var_2, xlim, ylim)

plt.show()
