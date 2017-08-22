import math
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

'''
主要用于解释MSE的偏差-方差分解，MLE虽然是无偏的，但是它方差大，MAP虽然是有偏的，但是它方差小，所以并不一定MLE就比MAP更优
特别是数据样本很小时，MAP往往比MLE更优，但是当数据样本趋近于无穷大时，MLE是渐近最优的估计器
'''

def Gaussian_Posterior(N, k0, theta_0, theta, sigma):
    w = N / (N + k0)
    E_x = w * theta + (1 - w) * theta_0
    var_x = w**2 * sigma**2 / N
    return E_x, var_x

def pdf(x, k0):
    N = 5
    theta_nature = 1.0
    sigma_nature = 1     # 根据图形估计出来的标准差
    theta_0 = 0          # 先验分布的均值
    
    mu, var = Gaussian_Posterior(N, k0, theta_0, theta_nature, sigma_nature)
    rv = ss.norm(mu, math.sqrt(var))  # 以高斯分布来画图
    
    return rv.pdf(x)

def odds(N, k0):
    theta_nature = 1.0
    sigma_nature = 1     # 根据图形估计出来的标准差
    theta_0 = 0          # 先验分布的均值
    
    MSE_MLE = sigma_nature**2 / N
    mu, var = Gaussian_Posterior(N, k0, theta_0, theta_nature, sigma_nature)
    print('mu: ', mu)
    print('var: ', var)
    MSE_Postmean = (mu - theta_nature)**2 + var
    
    return MSE_Postmean / MSE_MLE

x = np.linspace(-1, 2.5, 70)

plt.figure(figsize=(11, 5))
plt.subplot(121)
plt.xlim(-1, 2.5)
plt.ylim(0, 1.5)
plt.title('sampling distribution, truth = 1.0, prior = 0.0, n = 5', fontdict={'fontsize': 10})
plt.plot(x, pdf(x, 0), color='midnightblue', linestyle='-', marker='o', label='postMean0')
plt.plot(x, pdf(x, 1), color='red', linestyle='-', marker='x', label='postMean1')
plt.plot(x, pdf(x, 2), color='black', linestyle='-', marker='*', label='postMean2')
plt.plot(x, pdf(x, 3), color='green', linestyle='-', marker='>', label='postMean3')
plt.legend()

N = np.arange(1, 50, 2)
odds_1 = odds(N, 0)

plt.subplot(122)
plt.xlim(0, 50)
plt.ylim(0.5, 1.3)
plt.xlabel('sample size')
plt.ylabel('relative MSE')
plt.title('MSE of postmean / MSE of MLE', fontdict={'fontsize': 10})
plt.plot(N, odds(N, 0), color='midnightblue', linestyle='-', marker='o', label='postMean0')
plt.plot(N, odds(N, 1), color='red', linestyle='-', marker='x', label='postMean1')
plt.plot(N, odds(N, 2), color='black', linestyle='-', marker='*', label='postMean2')
plt.plot(N, odds(N, 3), color='green', linestyle='-', marker='>', label='postMean3')

plt.legend()
plt.show()


