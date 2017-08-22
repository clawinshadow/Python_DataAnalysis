import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

'''
使用Frequentist Risk Function MSE来比较几个estimator，虽然因为不知道真实的参数θ*，无法精确的计算每个估计器的值，但是
依然可以通过直观的比较，来得出某些估计器在任何情况下都比另外的一些要好，即MSE更小

x的抽样来自于真实分布 N(θ∗, σ2 = 1).

• δ1(x) = mean(x),     the sample mean
• δ2(x) = x~,          the sample median
• δ3(x) = θ0,          a fixed value
• δκ(x),               the posterior mean under a N(θ|θ0, σ2/κ) prior:
             N                 κ
  δκ(x) = -------*mean(x) + --------*θ0 = w*mean(x) + (1-w)θ0
           N + κ             N + κ
  考虑两个不同的先验 κ = 1, κ = 5

  MSE(δ1|θ*) = σ**2/N
  MSE(δ2|θ*) = π/2N
  MSE(δ3|θ*) = (θ∗ − θ0)**2
  
                   N * σ**2 + κ**2 * (θ0 − θ*)**2
  MSE(δκ|θ∗) = --------------------------------------
                          (N + κ)**2
'''

def postmean(sigma, N, theta_0, k, x):
    return (N * sigma**2 + k**2 * (x - theta_0)**2) / (N + k)**2

def Draw(index, N):
    theta_0 = 0                       # 根据书里面图上画的，θ0 = 0
    x = np.linspace(-2, 2, 500)       # x轴上是真实的均值θ∗，画出来的图形意味着对许多个不同的θ∗，各个估计器的MSE大小
    MSE_1 = 1 / N
    MSE_2 = np.pi/(2 * N)
    MSE_3 = (x - theta_0) ** 2
    MSE_4 = postmean(1, N, theta_0, 1, x)
    MSE_5 = postmean(1, N, theta_0, 5, x)

    plt.subplot(index)
    plt.title('risk functions for N = {0}'.format(N))
    plt.xlabel(r'$\theta_*$')
    plt.ylabel(r'$R(\theta_*, \delta)$')
    plt.xlim(-2, 2)
    plt.ylim(0, 0.5)
    plt.plot(x, np.tile(MSE_1, len(x)), color='midnightblue', linestyle='-', label='MLE')
    plt.plot(x, np.tile(MSE_2, len(x)), 'r:', label='median')
    plt.plot(x, MSE_3, 'k-.', label='fixed')
    plt.plot(x, MSE_4, 'g--', label='postmean 1')
    plt.plot(x, MSE_5, 'c-', label='postmean 5')

N1 = 5
N2 = 20
fig = plt.figure(figsize=(11, 5))
fig.canvas.set_window_title('riskFnGauss')
Draw(121, N1)
Draw(122, N2)

plt.legend()
plt.show()
