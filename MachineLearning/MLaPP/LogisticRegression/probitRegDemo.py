import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

def logistic(x):
    return 1 / (1 + np.exp(-x))

Lambda = (np.pi / 8) ** 0.5
x = np.linspace(-6, 6, 200)
sigm = logistic(x)
probit = ss.norm().cdf(Lambda * x)

fig = plt.figure()
fig.canvas.set_window_title('probitRegDemo')

plt.subplot()
plt.axis([-6, 6, 0, 1])
plt.xticks(np.arange(-6, 7, 2))
plt.yticks(np.arange(0, 1.01, 0.1))
plt.plot(x, sigm, 'r-', label='sigmoid')
plt.plot(x, probit, 'b--', label='probit')

plt.legend()
plt.show()
