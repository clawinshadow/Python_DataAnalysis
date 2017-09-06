import numpy as np
import matplotlib.pyplot as plt

'''
红色和蓝色的混合颜色，用不同的比例混合，可用于MixGaussian Model的EM算法等图形
'''

def getcolor(n):
    colors = []
    for i in range(n):
        r = 1 - i / n
        g = 0
        b = i / n
        colors.append([r, g, b])

    return colors

x = np.random.randint(1, 20 ,10)
y = np.random.randint(5, 70, 10)
colors = getcolor(len(x))
print('colors: \n', np.array(colors))

fig = plt.figure()
fig.canvas.set_window_title('MixColor_BlueRed')
plt.subplot()
plt.scatter(x, y, s=100, c=colors, marker='o', edgecolors='none')

plt.show()
