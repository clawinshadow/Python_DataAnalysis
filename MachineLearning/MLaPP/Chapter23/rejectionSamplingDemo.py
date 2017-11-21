import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

'''
当最简单的采样方法，使用cdf的反函数不可用时，还有另一种简单的替代方法，rejection sampling

Assumption of rejection sampling:
1. 我们想要采样的是概率分布 p(x), p~(x)是它的unormalized的分布，并且我们并不知道常量Zp, 即 p(x) = p~(x)/Zp
2. 我们知道另外一个概率分布 q(x), 满足 M*q(x) >= p~(x), 在整个定义域内都成立，并且我们可以很容易的对q(x)进行采样

steps of rejection sampling:
1. 先对q(x)进行采样
2. 再对U(0, 1)抽样一个u，如果 u*M*q(x) < p~(x)，则我们接受这个x，否则reject掉

本示例其实与这个方法没什么太大的关系，只是说明一下p(x)和q(x)之间的关系而已
'''

xs = np.linspace(0, 10, 200)
alpha = 5.7
lambda_ = 2
k = np.floor(alpha)
M = ss.gamma.pdf(alpha - k, a=alpha, loc=0, scale=1/lambda_) / ss.gamma.pdf(alpha - k, a=k, loc=0, scale=1/(lambda_-1))
print(M)
px = ss.gamma.pdf(xs, a=alpha, loc=0, scale=1/lambda_)
qx = M * ss.gamma.pdf(xs, k, 0, scale=1/(lambda_-1))

# plots
fig = plt.figure()
fig.canvas.set_window_title('rejectionSamplingDemo')

ax = plt.subplot()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(direction='in')
plt.axis([0, 10, 0, 1.4])
plt.xticks(np.linspace(0, 10, 6))
plt.yticks(np.linspace(0, 1.4, 8))
plt.plot(xs, px, '-', label='target p(x)', lw=2)
plt.plot(xs, qx, 'r:', label='comparison function Mq(x)', lw=2)
plt.legend()

plt.tight_layout()
plt.show()