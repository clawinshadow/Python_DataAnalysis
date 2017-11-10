import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from mcStatDist import *

# load data
data = sio.loadmat('harvard500.mat')
print(data.keys())  # G is the adjacent matrix, U is the related URL
G = data['G']
U = data['U']

# spy plot, especially for displaying the sparsity
fig = plt.figure(figsize=(10.5, 5))
fig.canvas.set_window_title('pagerankDemoPmtk')

ax = plt.subplot(121)
# default ticks of xaxis is on top, it's more convenient to display sparsity,
# that's fine, don't want to switch back to bottom
plt.spy(G, marker='.', markersize=1, color='midnightblue')
plt.xlabel('nz = 2636')

# use power method to get stationary distribution
pi = powermethod(G)
print(pi)

# plot the stationary distribution vector
plt.subplot(122)
plt.axis([-10, 510, 0, 0.02])
plt.xticks(np.linspace(0, 500, 6))
plt.yticks(np.linspace(0, 0.02, 11))
plt.bar(np.linspace(0, 499, 500), pi, color='midnightblue', edgecolor='midnightblue')

plt.tight_layout()
plt.show()