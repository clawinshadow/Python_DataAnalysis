import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

'''
poisson.pmf(k) = exp(-mu) * mu**k / k!

通俗点来说，参数k表示根据以往经验，某一段时间内发生某个事件的平均次数。那么接下来未来的同样一段时间内，
发生该事件的次数就是一个离散的随机变量，服从泊松分布。最常见的例子是交通事故的发生次数
'''

x = np.arange(0, 31, 1)
rv1 = ss.poisson(1)    # 服从参数为1的泊松分布 Poi(1)
rv2 = ss.poisson(10)
y1 = rv1.pmf(x)
y2 = rv2.pmf(x)

plt.figure(figsize=(11, 5))
plt.subplot(121)
plt.bar(x, y1, align='center')
plt.xlim(-1, 30)
plt.ylim(0, 0.4)
plt.title(r'$Poi(\lambda=1.0)$')

plt.subplot(122)
plt.bar(x, y2, align='center')
plt.xlim(-1, 30)
plt.ylim(0, 0.14)
plt.title(r'$Poi(\lambda=10.0)$')

plt.show()

