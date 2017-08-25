import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mm

def randRange(n, vmin, vmax):
    '''返回[vmin, vmax]之间的n个随机数，服从Uniform(vmin, vmax)'''
    return vmin + (vmax - vmin) * np.random.rand(n)

def Draw(color, marker, zlow, zhigh):
    xs = randRange(n, 23, 32)
    ys = randRange(n, 0, 100)
    zs = randRange(n, zlow, zhigh)
    ax.scatter(xs, ys, zs, c=color, marker=marker)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

n = 100
Draw('red', 'o', -50, -30)
Draw('blue', '^', -25, -5)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
