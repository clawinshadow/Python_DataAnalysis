import numpy as np
import matplotlib.pyplot as plt

'''
Simple demo of a scatter plot.
'''

N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N) # 每个散点的颜色
area = np.pi * (15 * np.random.rand(N))**2
print(area) # 每个圆点的面积

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
# splt.scatter(x, y, s=area, marker='s', c=colors, alpha=0.5) # square marker
plt.show()
