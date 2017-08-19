import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-2, 2, 0.01)
y1 = np.power(1 - (np.abs(x) - 1)**2, 1/2)
y2 = -3 * np.power(1 - (np.abs(x) / 2)**0.5, 1/2)

fig = plt.figure(figsize=(9, 7))
fig.canvas.set_window_title('FanFan & LanLan')

plt.fill_between(x, y2, y1, color='pink')
font = {'family': 'fantasy',
        'color':  'lightcoral',
        'weight': 'normal',
        'size': 16,
        }

plt.text(-2.2, -2.5, 'Fan & Lan stay together', fontdict=font)
plt.text(0.4, -2.5, 'forever', fontdict=font)
plt.arrow(2, 0.8, -0.22, -0.174, color='hotpink')
plt.arrow(0.03, -0.865, -1.36, -1.14, head_width=0.1, color='hotpink')
plt.xlim([-2.5, 2.5])
plt.ylim([-3, 1])
plt.axis('off')

plt.show()
