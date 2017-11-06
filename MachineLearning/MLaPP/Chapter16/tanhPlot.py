import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 201)
y1 = np.tanh(x)
y2 = 1 / (1 + np.exp(-x))

fig = plt.figure()
fig.canvas.set_window_title('tanhPlot')

plt.subplot()
plt.axis([-10, 10, -1, 1])
plt.xticks(np.linspace(-10, 10, 5))
plt.yticks(np.linspace(-1, 1, 11))
plt.plot(x, y1, 'r-', lw=2, label='tanh')
plt.plot(x, y2, 'g:', lw=2, label='sigmoid')
plt.legend()

plt.tight_layout()
plt.show()