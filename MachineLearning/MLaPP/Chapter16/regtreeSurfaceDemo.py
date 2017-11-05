import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x11 = 5
x12 = 3
x21 = 7
x22 = 3

X1, X2 = np.meshgrid(np.linspace(0, 10, 101), np.linspace(0, 10, 101))
r = np.linspace(2, 10, 5)

tree = np.zeros(X1.shape)
tree[(X1 <= x11) & (X2 <= x21)] = r[0]
tree[(X1 > x11) & (X2 <= x22)] = r[1]
tree[(X1 > x11) & (X2 > x22)] = r[2]
tree[(X1 <= min(x11, x22)) & (X2 > x21)] = r[3]
tree[(X1 <= x11) & (X1 > x12) & (X2 > x21)] = r[4]

fig = plt.figure()
fig.canvas.set_window_title('regtreeSurfaceDemo')

ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, tree, cmap='Paired')

plt.tight_layout()
plt.show()