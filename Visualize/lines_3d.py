import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mm

fig = plt.figure(figsize=(13, 4))
ax = fig.add_subplot(131, projection='3d')
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)
# 3d的plot方法里面的x, y 和 z都是长度相等的一维数组即可，不像contour里面需要meshgrid
ax.plot(x, y, z, label='parametric curve')

ax2 = fig.add_subplot(132, projection='3d')
ax2.plot(y, z, x, zdir='x', label='zdir = x')  # 以x的值为z轴，垂直的方向

ax3 = fig.add_subplot(133, projection='3d')
ax3.plot(x, z, y, zdir='y', label='zdir = y')  # 以y的值为z轴，垂直的方向

ax.legend()
ax2.legend()
ax3.legend()

plt.show()
