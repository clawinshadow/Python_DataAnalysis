import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

'''
这个用的比较多，写一下常用参数的含义：
X, Y, Z: 显然这是数据样本点的参数，这里的格式就需要跟contour()一样了，都是2D的数组，即meshgrid
rstride: 绘制表面的行间距
cstride: 绘制表面的列间距
rcount: 绘制表面的行上限
ccount: 绘制表面的列上限
color: 颜色
cmap: 渐变的颜色
....余下几个不常用的不列举出来了
'''

fig = plt.figure()
ax = fig.gca(projection='3d') # 除了add_subplot(111, projection='3d')的另一种方法

# Generate Data
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)  # 必须是meshgrid
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)             # X, Y, Z都是2D数组

# surf = ax.plot_surface(X, Y, Z, color='r')    # 表面全是红色
# surf = ax.plot_surface(X, Y, Z, cmap='jet')   # 表面是cmap定义的变化的颜色，Z值越高，颜色越暖，基本上
surf = ax.plot_surface(X, Y, Z, cmap='jet') # 抹去每个小方格之间的界线
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))  # 完全按照zlim来等分10个ticks
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))  # 每个tick保留两位小数

plt.show()
