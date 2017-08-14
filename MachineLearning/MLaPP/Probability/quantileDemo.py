import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 200)
y1 = ss.norm.cdf(x)  # ss.norm，不带任何参数，就是默认的标准正态分布 N(0, 1)

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title("cdf")
plt.ylim(0, 1)
plt.xlim(-3, 3)
plt.xticks(np.linspace(-3, 3, 7))
plt.plot(x, y1)

# alpha = 0.5
x2 = np.linspace(-4, 4, 200)
y2 = ss.norm.pdf(x2)
plt.subplot(122)
plt.plot(x2, y2)
plt.ylim([0, 0.5])
plt.xlim(-4, 4)
plt.xticks(np.linspace(-4, 4, 9))

left_quantile = ss.norm.ppf(0.025)
right_quantile = ss.norm.ppf(0.975)
x_left = np.linspace(-4, left_quantile, 200)
x_right = np.linspace(right_quantile, 4, 200)
# 本来是两个参数y1和y2，本例中只用到了y1，表示y1与坐标轴x之间的夹角区域
plt.fill_between(x_left, ss.norm.pdf(x_left), color='b')
plt.fill_between(x_right, ss.norm.pdf(x_right), color='b')
# xy表示箭头的坐标，xytext表示文本的坐标
plt.annotate(r'$\alpha/2$', xy=(left_quantile-0.5, ss.norm.pdf(left_quantile-0.5)),
             xytext=(-2.5, 0.1), arrowprops=dict(facecolor='k'))
plt.annotate(r'$1-\alpha/2$', xy=(right_quantile+0.5, ss.norm.pdf(right_quantile+0.5)),
             xytext=(2.5, 0.1), arrowprops=dict(facecolor='k'))

plt.show()
