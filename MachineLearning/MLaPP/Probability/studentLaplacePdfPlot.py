import math
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

x = np.linspace(-4, 4, 200)
gauss = ss.norm()
laplace = ss.laplace(0, 1/math.sqrt(2))
student = ss.t(1)  # 不能使用 t(0, 1, 1) 构造函数只有一个自由度。。。
y1 = gauss.pdf(x)
y2 = laplace.pdf(x)
y3 = student.pdf(x)
y1_log = gauss.logpdf(x)    # 开口向下的二次函数
y2_log = laplace.logpdf(x)  # 线性的，两条相交的直线
y3_log = student.logpdf(x)  # 依然是不规则非线性的


plt.figure(figsize=(11, 5))
plt.subplot(121)
plt.ylim(0, 0.8)
plt.xlim(-4, 4)
plt.plot(x, y1, ls=':', color='k', label='Gauss')
plt.plot(x, y2, ls='-', color='r', label='Laplace')
plt.plot(x, y3, ls='--', color='b', label='Student')
plt.legend()

plt.subplot(122)
plt.ylim(-9, 0)
plt.xlim(-4, 4)
plt.plot(x, y1_log, ls=':', color='k', label='Gauss')
plt.plot(x, y2_log, ls='-', color='r', label='Laplace')
plt.plot(x, y3_log, ls='--', color='b', label='Student')
plt.legend()

plt.show()
