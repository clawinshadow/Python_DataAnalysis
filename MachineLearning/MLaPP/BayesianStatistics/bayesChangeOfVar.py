import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

def y_func(x):
    return 1 / (1 + np.exp(5 - x))

rv1 = ss.norm(6, 1)              # x ~ N(6, 1), mode = 6
x_samples = rv1.rvs(10**6)       # 样本数量足够大才能足够的接近真实的分布

plt.figure()
plt.subplot()
plt.hist(x_samples, bins=100, normed=True, color='red', edgecolor='black')
plt.xlim(0, 12)
plt.ylim(0, 1)

x = np.linspace(0, 12, 1000)
y = y_func(x)
plt.plot(x, y, color='darkblue', linewidth=2)

y_samples = y_func(x_samples)
plt.hist(y_samples, bins=100, normed=True, orientation='horizontal', color='green', edgecolor='k')

x_mode = 6
y_point = y_func(x_mode)
plt.plot([x_mode, x_mode], [0, y_point], color='cyan', linewidth=2)
plt.plot([0, x_mode], [y_point, y_point], color='cyan', linewidth=2)

plt.text(8, 0.2, r'$P_x$')
plt.text(8, 0.9, 'g')
plt.text(1, 0.3, r'$P_y$')

plt.show()
