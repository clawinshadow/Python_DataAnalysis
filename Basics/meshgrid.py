import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab, cm

'''
about np.meshgrid.
https://docs.scipy.org/doc/numpy/reference/generated/numpy.meshgrid.html
其实说穿了就相当于在坐标系内织网，以2维坐标系为例，我们给定一组长度为 N 的X轴上的点 [x1, x2, ..., xn],
以及一组长度为 M 的Y轴上的点 [y1, y2, ..., ym], 那么我们可以想象从X轴上的每个点生长出垂直于X轴的直线，
从Y轴上生长出垂直于Y轴的直线，然后这些直线彼此相交，构成了一个网，然后每个交点就是节点，由meshgrid生成
节点的总数为 N * M 个，不强求必须得是正方形的网格，即 M = N

一般在构造二元函数，为了画等高图或者3D图形的时候经常使用这个函数，构建2维的自变量坐标点集
'''

x = np.array([1, 2, 3])
y = np.array([4, 5, 6, 7])
zx, zy = np.meshgrid(x, y)
# meshgrid会返回两个shape一样的矩阵，y的长度 乘以 x的长度，都是 4 * 3
# 第一个 是将X复制四次，垂直堆叠，第二个是将y转置，复制3次，水平堆叠
print('x: \n', x)
print('y: \n', y)
print('zx = np.meshgrid(x, y)[0]: \n', zx)
print('zy = np.meshgrid(x, y)[1]: \n', zy)

# 画图
x = np.arange(-5, 5, 0.01)
y = np.arange(-5, 5, 0.01)
xx, yy = np.meshgrid(x, y)
z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
levels = np.linspace(z.min(), z.max(), 100)
norm = cm.colors.Normalize(vmax=abs(z).max(), vmin=-abs(z).max())
plt.contourf(xx, yy, z, levels, colors='r', norm=norm)
plt.show()
