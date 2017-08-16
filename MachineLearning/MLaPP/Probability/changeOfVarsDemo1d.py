import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.subplot(131)
plt.hlines(0.5, -1, 1, color='darkblue') # 画水平线，参数分别是y, xmin, xmax, 每条线的长度可以不一样
plt.xticks([-1, 0, 1])
plt.yticks([-0.5, 0, 0.5, 1, 1.5])

def pdf_y(y):
    if np.any(y < 0) or np.any(y > 1):
        return np.nan
    else:
        return 0.5 * np.power(y, -0.5)   # x~Uni(-1, 1), y=x**2, 则p(y) = 1/2 * power(y, -1/2)

y = np.linspace(0.01, 1, 200)
probs = pdf_y(y)

plt.subplot(132)
plt.plot(y, probs, color='darkblue')
plt.xticks([0, 0.5, 1])
plt.yticks([0, 2, 4, 6])

# change of variables, in a MC way
# loc 表示区间的最小值，不是中间值，scale表示整个取值范围的大小
samples = ss.uniform(loc=-1, scale=2).rvs(1000)
y_samples = np.power(samples, 2)
weights = np.tile(1 / len(samples), len(samples))

plt.subplot(133)
plt.hist(y_samples, bins=30, weights=weights, color='darkblue', edgecolor='k')
plt.xticks([0, 0.5, 1])
plt.yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25])

plt.show()
