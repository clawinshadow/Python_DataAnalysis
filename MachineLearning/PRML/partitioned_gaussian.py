import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

'''
2. 关于高斯分布的条件分布和边际分布
   给定一个联合高斯分布 Ν(x|μ, Σ), 精度矩阵 Λ.inv = Σ
   
                 xa                          μa                   Σaa  Σab               Λaa  Λab
   将 x 划分为：      相应的均值向量μ划分为:       协方差矩阵为：            精度矩阵为：
                 xb                          μb                   Σba  Σbb               Λba  Λbb

   条件分布：
         p(xa|xb) = N(x|μa - Λaa.inv * Λab * (xb - μb), Λaa.inv)
         p(xa|xb) = N(x|μa + Σab * Σbb.inv * (xb - μb), Σaa - Σab * Σbb.inv * Σba)
         p(xb|xa) = N(x|μb - Λbb.inv * Λba * (xa - μa), Λbb.inv)
         p(xb|xa) = N(x|μb + Σba * Σaa.inv * (xa - μa), Σbb - Σba * Σaa.inv * Σab)
         
   边际分布
         p(xa) = N(x|μa, Σaa)
         p(xb) = N(x|μb, Σbb)
'''

# sample for partitioned gaussian distribution
x, y = np.mgrid[0:1:0.01, 0:1:0.01]
pos = np.dstack((x, y))               # 构造画等高线图的点集，shape = (100, 100, 2)
mean = np.array([0.5, 0.5])
cov = np.array([[0.2, 0.18],
                [0.18, 0.2]])
rv = ss.multivariate_normal(mean, cov)
probs = rv.pdf(pos)
print(probs.ravel().max())
levels = np.array([1.2, 1.55, 1.75])  # 只画三个圈，ps: 概率密度是可能大于一的

# visualize
fig = plt.figure(figsize=(11, 5))
fig.canvas.set_window_title('Partitioned Gaussian Distribution')
plt.subplot(121)
x2 = np.linspace(0, 1, 100)
y2 = np.tile(0.7, 100)
plt.xticks([0, 0.5, 1.0])    # 设置显示出来的横轴上的数值
plt.yticks([0, 0.5, 1.0])
plt.xlabel('Xa')
plt.ylabel('Xb')
# http://matplotlib.org/users/mathtext.html#mathtext-tutorial
plt.text(0.19, 0.72, r'$x_b = 0.7$', fontsize=15)
plt.contour(x, y, probs, levels, colors='limegreen')
plt.plot(x2, y2, 'r-')

# 因为只是一个二元的高斯分布，所以划分成两个标量
# 边际化分布 p(xa) ~ N(xa|0.5, 0.2)
# 条件分布 p(xa|xb) ~N(xa|0.5+(0.18/0.2)*(0.7-0.5), 0.2-0.18*0.18/0.2)
# conditional gaussian
mean_xa_xb = 0.5+(0.18/0.2)*(0.7-0.5)
cov_xa_xb = 0.2-0.18*0.18/0.2
mean_xa = 0.5
cov_xa = 0.2
rv1 = ss.norm(mean_xa, cov_xa)
rv2 = ss.norm(mean_xa_xb, cov_xa_xb)
x3 = np.linspace(0, 1, 200)
y_xa = rv1.pdf(x3)
y_xa_xb = rv2.pdf(x3)

plt.subplot(122)
plt.plot(x3, y_xa, 'b-')
plt.plot(x3, y_xa_xb, 'r-')
plt.xticks([0, 0.5, 1.0])
plt.yticks([0, 5, 10])
plt.xlabel('Xa')
plt.ylabel('Xb')
plt.text(0.3, 7, r'$P(x_a|x_b=0.7)$')
plt.text(0.15, 1.5, r'$P(x_a)$')

plt.show()
