import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def hinge(a):
    return np.max(np.c_[a, np.zeros(len(a))], axis=1)

x = np.linspace(0, 20, 201)
y1 = hinge(x - 5)
y2 = hinge(5 - x)
y3 = 25 - 4 * hinge(x - 5) + 20 * hinge(5 - x)

xx, yy = np.meshgrid(np.linspace(0, 20, 101), np.linspace(0, 20, 101))
x1, x2 = xx.ravel(), yy.ravel()
y4 = 2 - 2 * hinge(x1 - 5) + 3 * hinge(5 - x1) - hinge(x2 - 10) * hinge(5 - x1) +\
    -1.2 * hinge(10 - x2) * hinge(5 - x1)
y4 = y4.reshape(xx.shape)

# plots
fig = plt.figure(figsize=(13, 4))
fig.canvas.set_window_title('marsDemo')

ax1 = plt.subplot(131)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
plt.axis([-2, 22, -1, 16])
plt.xticks(np.linspace(0, 20, 5))
plt.yticks(np.linspace(0, 15, 4))
plt.plot(x, y1, color='midnightblue', linestyle='-', lw=2)
plt.plot(x, y2, 'r:', lw=2)

ax2 = plt.subplot(132)
plt.axis([-2, 22, -42, 130])
plt.xticks(np.linspace(0, 20, 5))
plt.yticks(np.linspace(-40, 120, 9))
plt.plot(x, y3, color='midnightblue', linestyle='-', lw=2)

ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(xx, yy, y4, cmap='jet', edgecolors='k')

plt.tight_layout()
plt.show()
