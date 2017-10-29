import numpy as np
import matplotlib.pyplot as plt

err = np.arange(-3, 3.01, 0.1)
epsilon = 1.5
indices = np.abs(err) <= epsilon

L2 = err**2
huber = 0.5 * indices * err**2 + (1 - indices) * (epsilon * (np.abs(err) - epsilon / 2))
vapnik = indices * 0 + (1 - indices) * (np.abs(err) - epsilon)

plt.figure()
plt.subplot()
plt.axis([-3, 3, -0.5, 5])
plt.xticks(np.linspace(-3, 3, 7))
plt.yticks(np.linspace(-0.5, 5, 12))
plt.plot(err, L2, 'r-', lw=2, label='L2')
plt.plot(err, vapnik, 'b:', lw=2, label=r'$\epsilon-insensitive$')
plt.plot(err, huber, 'g-.', lw=2, label='huber')

plt.legend()
plt.tight_layout()
plt.show()