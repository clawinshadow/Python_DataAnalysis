import numpy as np
import matplotlib.pyplot as plt

p = np.linspace(1e-5, 0.9999, 101)
gini = 2 * p * (1 - p)
entropy = -p * np.log(p) - (1 - p) * np.log(1 - p)
err = 1 - np.max(np.c_[p, 1 - p], axis=1)

# rescale for plot
entropy = 0.5 * entropy / np.max(entropy)

# plots
fig = plt.figure()
fig.canvas.set_window_title('giniDemo')

plt.subplot()
plt.axis([0, 1, 0, 0.5])
plt.xticks(np.linspace(0, 1, 6))
plt.yticks(np.linspace(0, 0.5, 11))
plt.plot(p, err, 'g-', lw=2, label='Error Rate')
plt.plot(p, gini, 'b:', lw=2, label='Gini')
plt.plot(p, entropy, 'r--', lw=2, label='Entropy')
plt.legend()

plt.tight_layout()
plt.show()